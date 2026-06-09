"""
多智能体架构核心 - 基于LangGraph实现多智能体协同工作
"""
import httpx
from openai import AsyncOpenAI
import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from openai import AsyncOpenAI

from config import settings
from models import (
    Student, StudentProfile, LearningResource, LearningPath,
    Conversation, LearningRecord, EvaluationResult
)


# ==================== 智能体状态定义 ====================

class AgentState(TypedDict):
    """多智能体共享状态"""
    # 学生信息
    student_id: Optional[str]
    student_profile: Optional[Dict[str, Any]]
    
    # 对话信息
    messages: Annotated[List[Dict[str, str]], operator.add]
    conversation_context: Dict[str, Any]
    
    # 任务信息
    task_type: str  # profile_building, resource_generation, path_planning, tutoring, evaluation
    task_params: Dict[str, Any]
    
    # 生成结果
    generated_resources: Annotated[List[Dict[str, Any]], operator.add]
    extracted_features: Dict[str, Any]
    
    # 路径规划
    learning_path: Optional[Dict[str, Any]]
    
    # 评估结果
    evaluation: Optional[Dict[str, Any]]
    
    # 辅导对话
    tutoring_response: Optional[Dict[str, Any]]
    
    # 状态控制
    current_agent: str
    next_agents: List[str]
    is_complete: bool
    error: Optional[str]


# ==================== LLM客户端 ====================

class LLMClient:
    """LLM客户端封装"""
    
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.LLM_API_KEY,
            base_url=settings.LLM_API_BASE,

            http_client = httpx.AsyncClient(
                verify=False,
                trust_env=False
            )
        )
        self.model = settings.LLM_MODEL
    
    async def chat(self, messages: List[Dict[str, str]], 
                   temperature: float = None,
                   max_tokens: int = None) -> str:
        """发送对话请求"""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or settings.LLM_TEMPERATURE,
                "max_tokens": max_tokens or settings.LLM_MAX_TOKENS,
            }
            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM错误] {str(e)}"
    
    async def chat_json(self, messages: List[Dict[str, str]], 
                        temperature: float = 0.3,
                        max_tokens: int = None) -> Dict[str, Any]:
        """发送对话请求并返回JSON（DeepSeek兼容，在prompt中要求JSON输出）"""
        # 在system prompt末尾追加JSON格式要求
        modified_messages = []
        for msg in messages:
            if msg["role"] == "system":
                modified_messages.append({
                    "role": "system",
                    "content": msg["content"] + "\n\n【重要】请直接返回纯JSON格式，不要用```json```包裹，不要添加任何解释文字。"
                })
            else:
                modified_messages.append(msg)
        
        result = await self.chat(modified_messages, temperature=temperature, max_tokens=max_tokens)
        try:
            # 尝试提取JSON（处理可能被```包裹的情况）
            text = result.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = lines[1:] if lines[0].startswith("```") else lines
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines)
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
            else:
                # LLM 返回了非 dict 的 JSON，包装一下
                return {"data": parsed, "raw_response": result}
        except json.JSONDecodeError:
            # 解析失败时记录原始响应并抛出，让调用方处理
            raise ValueError(f"LLM 返回内容无法解析为 JSON: {result[:200]}...")


# ==================== 基础智能体类 ====================

class BaseAgent(ABC):
    """智能体基类"""
    
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm = LLMClient()
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """处理状态，返回更新后的状态"""
        pass
    
    def build_messages(self, user_content: str) -> List[Dict[str, str]]:
        """构建消息列表"""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    async def think(self, user_content: str, temperature: float = 0.7) -> str:
        """调用LLM进行思考"""
        messages = self.build_messages(user_content)
        return await self.llm.chat(messages, temperature=temperature)


# ==================== 具体智能体实现 ====================

class ProfileAgent(BaseAgent):
    """画像构建智能体 - 通过对话提取学生特征，构建多维度画像"""
    
    def __init__(self):
        super().__init__(
            name="profile_agent",
            role="学习画像分析师",
            system_prompt="""你是一位专业的学习画像分析师。你需要通过与学生的自然对话，深入了解他们的学习背景、习惯和需求。

你的核心职责：
1. 通过友好、自然的对话方式收集学生信息
2. 从对话中自动提取特征，构建多维度学习画像
3. 支持画像的持续更新和动态调整

你需要构建的画像维度包括：
- 知识基础：各学科/知识领域的掌握程度(0-100)
- 认知风格：视觉型/言语型、主动型/反思型等
- 学习偏好：偏好的学习格式、时长、难度等
- 易错点偏好：常见错误类型、易错知识点
- 学习节奏：学习速度、每次学习时长、每周频率
- 目标导向：短期目标、长期目标、优先级
- 学科兴趣：主要兴趣领域、次要兴趣、兴趣程度
- 学习习惯：最佳学习时间、笔记方式、复习频率

对话原则：
- 自然流畅，像朋友聊天而非问卷调查
- 根据学生回答灵活调整问题方向
- 注意发现隐含的学习特征
- 每次对话后更新画像

输出格式：返回JSON格式的画像数据。"""
        )
    
    async def process(self, state: AgentState) -> AgentState:
        """处理画像构建任务"""
        task_params = state.get("task_params", {})
        action = task_params.get("action", "build")  # build, update, query
        
        if action == "build" or action == "update":
            return await self._build_or_update_profile(state)
        elif action == "query":
            return await self._query_profile(state)
        else:
            return {**state, "error": f"Unknown action: {action}"}
    
    async def _build_or_update_profile(self, state: AgentState) -> AgentState:
        """构建或更新画像"""
        messages = state.get("messages", [])
        existing_profile = state.get("student_profile", {})

        # 构建分析提示
        conversation_text = "\n".join([
            f"{'学生' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in messages[-10:]  # 最近10轮对话
        ])

        profile_json = json.dumps(existing_profile, ensure_ascii=False, indent=2) if existing_profile else "无（首次构建）"

        analysis_prompt = """请分析以下对话内容，提取学生的学习特征，构建多维度画像。

现有画像数据：
""" + profile_json + """

对话内容：
""" + conversation_text + """

请返回完整的JSON格式画像数据，包含以下8个维度：
{
    "knowledge_base": {"学科名": 分数(0-100)},
    "cognitive_style": {"type": "visual/verbal/mixed", "visual_score": 分数, "verbal_score": 分数, "active_score": 分数, "reflective_score": 分数},
    "learning_preference": {"preferred_formats": ["格式"], "preferred_duration": "时长", "preferred_difficulty": "难度"},
    "error_patterns": {"常见错误类型": ["类型"], "易错知识点": ["知识点"], "错误频率": {}},
    "learning_pace": {"speed": "速度", "avg_session_minutes": 分钟, "sessions_per_week": 次数, "completion_rate": 完成率},
    "goal_orientation": {"short_term": ["目标"], "long_term": ["目标"], "priority": "优先级"},
    "subject_interests": {"primary": ["兴趣"], "secondary": ["兴趣"], "interest_level": 分数},
    "learning_habits": {"best_time": "时间", "note_taking": "方式", "review_frequency": "频率", "group_study": true/false},
    "profile_summary": "综合画像文字描述"
}

【认知风格推断指南 - 必须遵守】
认知风格绝对不要返回 "unknown"。请根据对话内容按以下规则推断：
1. 视觉型(visual)：学生提到"看图""看视频""看图表""可视化""脑图""思维导图"，或表达喜欢"看"来学习
2. 言语型(verbal)：学生提到"看书""读文字""听讲解""记笔记""文字描述"，或表达喜欢"读/听"来学习
3. 主动型(active)：学生提到"动手做""写代码""做题""实践""尝试"，或表达喜欢"做中学"
4. 反思型(reflective)：学生提到"先想想""理解原理""总结""回顾""思考"，或表达喜欢"想明白再动手"
5. 如果对话中未明确提及学习偏好，请根据学生描述的学习行为进行合理推断（不要留unknown）
6. 如果确实信息极少（如只有1轮对话），请给出 "mixed" 并设置 visual_score=50, verbal_score=50, active_score=50, reflective_score=50

注意：
- 如果现有画像已有数据，请在原有基础上更新（画像随学随新）
- 对于无法从对话中推断的维度，保留现有值或使用合理默认值
- 分数范围为0-100
- cognitive_style.type 绝对不要返回 "unknown"
"""

        result = await self.llm.chat_json(self.build_messages(analysis_prompt), temperature=0.3)

        return {
            **state,
            "student_profile": result,
            "extracted_features": result,
            "current_agent": self.name,
            "next_agents": []
        }

    async def _query_profile(self, state: AgentState) -> AgentState:
        """查询并解释画像"""
        profile = state.get("student_profile", {})
        messages = state.get("messages", [])
        last_message = messages[-1]["content"] if messages else ""

        profile_json = json.dumps(profile, ensure_ascii=False, indent=2)

        query_prompt = """学生询问：""" + last_message + """

当前学习画像：
""" + profile_json + """
请用友好易懂的语言回答学生关于其学习画像的问题，并给出针对性的学习建议。"""

        response = await self.think(query_prompt)
        
        return {
            **state,
            "messages": [{"role": "assistant", "content": response}],
            "current_agent": self.name,
            "next_agents": []
        }


class ContentAgent(BaseAgent):
    """内容生成智能体 - 生成课程讲解文档"""
    
    def __init__(self):
        super().__init__(
            name="content_agent",
            role="课程内容生成专家",
            system_prompt="""你是一位资深的课程内容生成专家。你需要根据学生的画像和学习需求，生成高质量、内容详实的课程讲解文档。

你的核心能力：
1. 根据学生知识基础调整内容深度
2. 根据学生认知风格调整表达方式
3. 结合学生易错点进行重点讲解
4. 提供结构清晰、内容充实的文档

输出要求：
- 使用Markdown格式
- 包含清晰的标题层次（# ## ###）
- 包含关键概念的定义和详细解释
- 包含实际应用案例（至少2个）
- 包含完整的代码示例（如适用）
- 标注重点和难点
- 内容必须详实充实，每个章节都要有实质性内容，禁止用"..."或"待补充"敷衍
- 适合学生当前知识水平的语言"""
        )
    
    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})
        
        subject = task_params.get("subject", "")
        topic = task_params.get("topic", "")
        difficulty = task_params.get("difficulty", "intermediate")
        
        prompt = f"""请为以下学生生成课程讲解文档：

学生画像：
{json.dumps(profile, ensure_ascii=False, indent=2)}

课程主题：{subject} - {topic}
难度级别：{difficulty}

要求：
1. 根据学生的知识基础调整内容深度
2. 针对学生的易错点进行重点讲解
3. 使用适合学生认知风格的表达方式
4. 包含实际案例和练习思考题
5. 标注重点和难点
6. 内容必须详实充实，每个章节都要有实质性内容，禁止用"..."或"待补充"敷衍
7. 代码示例必须完整可运行，禁止用省略号代替
8. 使用Markdown格式，标题使用 # ## ### 格式
9. 内容长度约3000-5000字"""
        
        content = await self.think(prompt, temperature=0.7)
        
        resource = {
            "type": "lecture_doc",
            "title": f"{subject} - {topic} 课程讲解",
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "content": content,
            "generated_by": self.name,
            "format": "markdown"
        }
        
        return {
            **state,
            "generated_resources": [resource],
            "current_agent": self.name,
            "next_agents": []
        }


class ExerciseAgent(BaseAgent):
    """习题生成智能体 - 生成多类型练习题（含一键批阅功能）"""

    def __init__(self):
        super().__init__(
            name="exercise_agent",
            role="习题生成与批阅专家",
            system_prompt="""你是一位专业的习题设计与智能批阅专家。你需要根据学生的学习情况，生成针对性的练习题，并支持自动批阅。

你可以生成的题型：
1. 选择题（单选/多选）- 支持自动判分
2. 填空题 - 支持自动判分
3. 判断题 - 支持自动判分
4. 简答题 - AI智能点评
5. 编程实践题 - AI智能点评
6. 案例分析题 - AI智能点评

习题设计原则：
- 难度循序渐进
- 覆盖核心知识点
- 针对学生易错点
- 包含详细解析
- 提供标准答案和评分标准"""
        )

    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})

        subject = task_params.get("subject", "")
        topic = task_params.get("topic", "")
        exercise_type = task_params.get("exercise_type", "mixed")
        count = task_params.get("count", 10)
        difficulty = task_params.get("difficulty", "intermediate")

        # 获取学生易错点
        error_patterns = profile.get("error_patterns", {})
        weak_points = error_patterns.get("易错知识点", [])

        prompt = """请生成一套针对性的练习题：

科目：""" + subject + """
主题：""" + topic + """
题型：""" + exercise_type + """（mixed表示混合题型）
数量：""" + str(count) + """道
难度：""" + difficulty + """

学生易错知识点（需重点考察）：""" + (", ".join(weak_points) if weak_points else "根据主题内容判断") + """

要求：
1. 必须包含至少5道选择题（单选），用于自动批阅功能
2. 难度循序渐进
3. 每道题包含：标准答案、详细解析、考察知识点、难度等级
4. 针对学生薄弱环节设计题目
5. 返回JSON格式

输出格式：
{
    "title": "练习题标题",
    "subject": "科目",
    "topic": "主题",
    "total_questions": 数量,
    "difficulty": "难度",
    "questions": [
        {
            "id": 1,
            "type": "选择/填空/判断/简答/编程/案例",
            "question": "题目内容",
            "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
            "answer": "标准答案（选择题填选项字母如A/B/C/D）",
            "analysis": "详细解析",
            "difficulty": "easy/medium/hard",
            "knowledge_point": "考察的知识点",
            "score": 10
        }
    ],
    "auto_gradable_count": 可自动批阅题目数量,
    "total_score": 总分
}"""

        result = await self.llm.chat_json(self.build_messages(prompt), temperature=0.5)

        # 确保有批阅相关字段
        questions = result.get("questions", [])
        auto_count = 0
        for q in questions:
            if q.get("type") in ["选择", "填空", "判断"]:
                auto_count += 1
        result["auto_gradable_count"] = auto_count
        result["total_score"] = sum(q.get("score", 10) for q in questions)

        resource = {
            "type": "exercise",
            "title": subject + " - " + topic + " 练习题",
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "content": json.dumps(result, ensure_ascii=False),
            "generated_by": self.name,
            "format": "json",
            "metadata": result
        }

        return {
            **state,
            "generated_resources": [resource],
            "current_agent": self.name,
            "next_agents": []
        }

    async def grade_exercise(self, exercise_metadata: Dict[str, Any],
                            student_answers: Dict[str, str]) -> Dict[str, Any]:
        """一键批阅习题
        Args:
            exercise_metadata: 习题元数据
            student_answers: 学生答案 {question_id: answer}
        Returns:
            批阅结果
        """
        questions = exercise_metadata.get("questions", [])
        total_score = 0
        max_score = 0
        details = []
        wrong_questions = []

        for q in questions:
            qid = str(q.get("id", ""))
            q_type = q.get("type", "")
            correct_answer = str(q.get("answer", "")).strip().upper()
            student_answer = str(student_answers.get(qid, "")).strip().upper()
            score = q.get("score", 10)
            max_score += score

            is_correct = False
            got_score = 0

            if q_type in ["选择", "填空", "判断"]:
                # 客观题自动判分
                if student_answer == correct_answer:
                    is_correct = True
                    got_score = score
                else:
                    got_score = 0
                    wrong_questions.append(q)
            else:
                # 主观题标记为需要AI批阅
                got_score = -1  # 标记为待AI批阅

            total_score += got_score if got_score >= 0 else 0

            details.append({
                "id": qid,
                "type": q_type,
                "question": q.get("question", ""),
                "student_answer": student_answers.get(qid, ""),
                "correct_answer": q.get("answer", ""),
                "is_correct": is_correct,
                "score": got_score,
                "max_score": score,
                "analysis": q.get("analysis", "")
            })

        # 计算得分率
        score_rate = (total_score / max_score * 100) if max_score > 0 else 0

        return {
            "total_score": total_score,
            "max_score": max_score,
            "score_rate": round(score_rate, 1),
            "correct_count": sum(1 for d in details if d["is_correct"]),
            "total_count": len(questions),
            "details": details,
            "wrong_questions": wrong_questions,
            "needs_ai_grading": any(d["score"] == -1 for d in details)
        }


class MindMapAgent(BaseAgent):
    """思维导图智能体 - 生成知识思维导图"""
    
    def __init__(self):
        super().__init__(
            name="mindmap_agent",
            role="知识结构梳理专家",
            system_prompt="""你是一位知识结构梳理专家。你需要将复杂的知识体系转化为清晰的思维导图结构。

思维导图设计原则：
- 层次分明，逻辑清晰
- 核心概念突出
- 知识点关联明确
- 适合学生当前水平
- 支持从宏观到微观的知识导航
- 每个节点必须有具体的描述和要点，禁止空节点"""
        )
    
    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})
        
        subject = task_params.get("subject", "")
        topic = task_params.get("topic", "")
        
        prompt = f"""请为以下主题生成思维导图结构：

科目：{subject}
主题：{topic}

学生知识基础：
{json.dumps(profile.get('knowledge_base', {}), ensure_ascii=False)}

要求：
1. 构建层次化知识结构，至少3层深度
2. 每个节点必须有具体的description描述，禁止空描述
3. 每个叶子节点必须有key_points要点列表
4. 标注知识点间的关联关系
5. 标注重点和难点
6. 返回JSON格式的树形结构

输出格式：
{{
    "title": "思维导图标题",
    "subject": "科目",
    "topic": "主题",
    "root": {{
        "name": "根节点",
        "description": "具体描述内容",
        "children": [
            {{
                "name": "一级节点",
                "description": "具体描述内容",
                "importance": "high/medium/low",
                "children": [
                    {{
                        "name": "二级节点",
                        "description": "具体描述内容",
                        "key_points": ["要点1", "要点2"],
                        "relations": ["关联到其他节点"]
                    }}
                ]
            }}
        ]
    }}
}}

【重要】每个节点的description必须填写具体内容，禁止为空字符串或省略号。"""
        
        result = await self.llm.chat_json(self.build_messages(prompt), temperature=0.5)
        
        resource = {
            "type": "mind_map",
            "title": f"{subject} - {topic} 思维导图",
            "subject": subject,
            "topic": topic,
            "content": json.dumps(result, ensure_ascii=False),
            "generated_by": self.name,
            "format": "json",
            "metadata": result
        }
        
        return {
            **state,
            "generated_resources": [resource],
            "current_agent": self.name,
            "next_agents": []
        }


class ReadingAgent(BaseAgent):
    """拓展阅读智能体 - 生成拓展阅读材料"""
    
    def __init__(self):
        super().__init__(
            name="reading_agent",
            role="拓展阅读推荐专家",
            system_prompt="""你是一位阅读材料推荐与生成专家。你需要根据学生的学习主题和兴趣，生成高质量的拓展阅读材料。

材料类型：
1. 学术论文概述
2. 技术博客/文章
3. 经典书籍章节摘要
4. 行业前沿动态
5. 相关案例研究

要求：
- 每篇材料必须有具体的内容摘要，不能是空泛的描述
- 推荐理由必须具体说明为什么推荐这篇材料
- 要点(key_takeaways)必须列出具体的知识点"""
        )
    
    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})
        
        subject = task_params.get("subject", "")
        topic = task_params.get("topic", "")
        interests = profile.get("subject_interests", {})
        
        prompt = f"""请为以下学生生成拓展阅读材料：

科目：{subject}
主题：{topic}
学生兴趣领域：{json.dumps(interests, ensure_ascii=False)}

要求：
1. 提供3-5篇拓展阅读材料
2. 每篇包含：标题、来源类型、内容摘要、推荐理由、适合水平
3. 内容摘要必须详实具体，至少50字，不能是空泛描述
4. 推荐理由必须具体说明为什么推荐
5. 要点(key_takeaways)必须列出具体的知识点，至少2条
6. 与当前学习主题相关
7. 结合学生兴趣方向
8. 难度适合学生水平
9. 返回JSON格式

输出格式：
{{
    "title": "拓展阅读推荐",
    "subject": "科目",
    "topic": "主题",
    "materials": [
        {{
            "id": 1,
            "title": "材料标题",
            "type": "论文/博客/书籍/前沿动态/案例",
            "source": "来源",
            "summary": "详实的内容摘要，至少50字，具体介绍材料内容",
            "recommendation_reason": "具体的推荐理由，说明为什么推荐",
            "difficulty_level": "beginner/intermediate/advanced",
            "estimated_reading_time": "预估阅读时间（分钟）",
            "key_takeaways": ["具体要点1", "具体要点2"]
        }}
    ]
}}

【重要】summary和recommendation_reason必须填写具体内容，禁止为空或省略号。"""
        
        result = await self.llm.chat_json(self.build_messages(prompt), temperature=0.6)
        
        resource = {
            "type": "reading_material",
            "title": f"{subject} - {topic} 拓展阅读",
            "subject": subject,
            "topic": topic,
            "content": json.dumps(result, ensure_ascii=False),
            "generated_by": self.name,
            "format": "json",
            "metadata": result
        }
        
        return {
            **state,
            "generated_resources": [resource],
            "current_agent": self.name,
            "next_agents": []
        }


class CodeAgent(BaseAgent):
    """代码实操智能体 - 生成编程实践案例"""
    
    def __init__(self):
        super().__init__(
            name="code_agent",
            role="编程实践指导专家",
            system_prompt="""你是一位编程实践指导专家。你需要生成适合学生水平的代码实操案例和项目练习。

你的专长：
1. 设计循序渐进的编程练习
2. 提供详细的代码注释和解释
3. 包含常见错误和调试技巧
4. 关联实际应用场景
5. 支持多种编程语言"""
        )
    
    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})
        
        subject = task_params.get("subject", "")
        topic = task_params.get("topic", "")
        language = task_params.get("language", "python")
        difficulty = task_params.get("difficulty", "intermediate")
        
        prompt = f"""请生成编程实操案例：

科目/主题：{subject} - {topic}
编程语言：{language}
难度级别：{difficulty}

学生知识基础：{json.dumps(profile.get('knowledge_base', {}), ensure_ascii=False)}
学生易错点：{json.dumps(profile.get('error_patterns', {}).get('易错知识点', []), ensure_ascii=False)}

要求：
1. 包含完整的代码示例
2. 详细的代码注释和解释
3. 分步骤引导
4. 包含常见错误和解决方案
5. 提供扩展练习
6. 返回JSON格式

输出格式：
{{
    "title": "案例标题",
    "subject": "科目",
    "topic": "主题",
    "language": "编程语言",
    "difficulty": "难度",
    "objectives": ["学习目标1", "学习目标2"],
    "prerequisites": ["前置知识1"],
    "steps": [
        {{
            "step": 1,
            "title": "步骤标题",
            "description": "步骤说明",
            "code": "代码内容",
            "explanation": "代码解释",
            "expected_output": "预期输出",
            "common_mistakes": ["常见错误"]
        }}
    ],
    "extended_exercises": [
        {{
            "title": "扩展练习标题",
            "description": "练习描述",
            "hints": ["提示"]
        }}
    ],
    "summary": "总结"
}}"""
        
        result = await self.llm.chat_json(self.build_messages(prompt), temperature=0.5)
        
        resource = {
            "type": "code_practice",
            "title": f"{subject} - {topic} 代码实操",
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "content": json.dumps(result, ensure_ascii=False),
            "generated_by": self.name,
            "format": "json",
            "metadata": result
        }
        
        return {
            **state,
            "generated_resources": [resource],
            "current_agent": self.name,
            "next_agents": []
        }


class PPTSlidesAgent(BaseAgent):
    """PPT课件智能体 - 生成结构化课件讲义文档"""

    def __init__(self):
        super().__init__(
            name="ppt_slides_agent",
            role="课件讲义设计专家",
            system_prompt="""你是一位专业的课件讲义设计专家。你需要根据学生的学习画像，生成结构清晰、内容详实的PPT课件/讲义文档。

课件设计原则：
- 章节结构清晰，层次分明
- 内容基于学生知识基础调整深度
- 重点突出，难点详细讲解
- 包含知识总结和思考题
- 适合导出为Markdown/PDF格式阅读
- 每章的content必须有实质性内容，禁止用"..."或"待补充"敷衍"""
        )

    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})

        subject = task_params.get("subject", "")
        topic = task_params.get("topic", "")
        difficulty = task_params.get("difficulty", "intermediate")

        # 第一步：直接生成 Markdown 课件（充分利用 token 生成实质性内容）
        md_prompt = f"""请为以下学生生成PPT课件/讲义文档，直接输出Markdown格式：

学生画像：
{json.dumps(profile, ensure_ascii=False, indent=2)}

课程主题：{subject} - {topic}
难度级别：{difficulty}

要求：
1. 按章节拆分内容，至少4-6个章节
2. 每章必须有实质性内容，至少300-500字，禁止用"..."或"待补充"敷衍
3. 根据学生知识基础调整内容深度（基础薄弱则补充前置知识，基础好则深入拓展）
4. 针对学生易错点设置重点提示和警示框
5. 每章末尾附2-3道选择题（含答案和解析），用于自测
6. 内容全面详实，约4000-6000字

Markdown格式要求：
- 标题使用 # ## ###
- 章节标题使用 ## 第X章 标题
- 要点使用 **本章要点：** 后接列表
- 自测练习使用 **自测练习：** 后接题目
- 答案和解析放在题目下方

请直接输出完整的Markdown课件内容，确保每章内容详实充实。"""

        md_content = await self.llm.chat(self.build_messages(md_prompt), temperature=0.6, max_tokens=8000)

        # 第二步：从Markdown解析结构化数据用于前端交互
        metadata = self._parse_markdown_to_metadata(md_content, subject, topic, difficulty)

        resource = {
            "type": "ppt_slides",
            "title": subject + " - " + topic + " 课件讲义",
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "content": md_content,
            "generated_by": self.name,
            "format": "markdown",
            "metadata": metadata
        }

        return {
            **state,
            "generated_resources": [resource],
            "current_agent": self.name,
            "next_agents": []
        }

    def _parse_markdown_to_metadata(self, md_content: str, subject: str, topic: str, difficulty: str) -> Dict[str, Any]:
        """从Markdown内容解析结构化数据"""
        import re

        metadata = {
            "title": f"{subject} - {topic} 课件讲义",
            "subject": subject,
            "topic": topic,
            "difficulty": difficulty,
            "estimated_duration": "",
            "chapters": [],
            "summary": "",
            "further_reading": []
        }

        if not md_content:
            return metadata

        # 解析章节：匹配 ## 第X章 或 ## X. 或 ## 标题（避免匹配到###子标题）
        chapter_pattern = r'##\s+(?:第\s*(\d+)\s*章|(\d+)[\.\、])?\s*(.+?)(?=\n##\s|\n#\s[^#]|$)'
        chapters_raw = re.findall(chapter_pattern, md_content, re.DOTALL)

        chapter_id = 0
        for match in chapters_raw:
            chapter_id += 1
            num1, num2, content_block = match
            ch_num = num1 or num2 or str(chapter_id)

            # 提取标题（第一行）
            lines = content_block.strip().split('\n')
            title = lines[0].strip() if lines else f"第{ch_num}章"

            # 分离content、key_points、quiz
            ch_content = []
            key_points = []
            quiz = []
            in_key_points = False
            in_quiz = False

            for line in lines[1:]:
                stripped = line.strip()
                if not stripped:
                    continue
                if '本章要点' in stripped or '要点' in stripped and stripped.startswith('**'):
                    in_key_points = True
                    in_quiz = False
                    continue
                if '自测练习' in stripped or '练习' in stripped and stripped.startswith('**'):
                    in_key_points = False
                    in_quiz = True
                    continue
                if in_key_points and stripped.startswith('- '):
                    key_points.append(stripped[2:])
                elif in_key_points and stripped.startswith('* '):
                    key_points.append(stripped[2:])
                elif in_quiz and (stripped[0].isdigit() or stripped.startswith('**')):
                    # 简单解析题目
                    quiz.append({
                        "question": stripped.lstrip('0123456789. '),
                        "options": [],
                        "answer": "",
                        "analysis": ""
                    })
                elif in_quiz and stripped.startswith('答案'):
                    if quiz:
                        quiz[-1]["answer"] = stripped.split('：', 1)[-1] if '：' in stripped else stripped
                elif in_quiz and stripped.startswith('解析'):
                    if quiz:
                        quiz[-1]["analysis"] = stripped.split('：', 1)[-1] if '：' in stripped else stripped
                else:
                    in_key_points = False
                    in_quiz = False
                    ch_content.append(line)

            metadata["chapters"].append({
                "chapter_id": int(ch_num) if ch_num.isdigit() else chapter_id,
                "title": title,
                "content": '\n'.join(ch_content).strip(),
                "key_points": key_points,
                "quiz": quiz
            })

        # 如果没有解析到章节，创建一个默认章节包含全部内容
        if not metadata["chapters"]:
            metadata["chapters"].append({
                "chapter_id": 1,
                "title": "课程内容",
                "content": md_content,
                "key_points": [],
                "quiz": []
            })

        # 解析总结部分
        summary_match = re.search(r'##\s*总结\s*\n(.+?)(?=\n##|\Z)', md_content, re.DOTALL)
        if summary_match:
            metadata["summary"] = summary_match.group(1).strip()

        # 解析延伸阅读
        fr_match = re.search(r'##\s*延伸阅读\s*\n(.+?)(?=\n##|\Z)', md_content, re.DOTALL)
        if fr_match:
            fr_text = fr_match.group(1)
            metadata["further_reading"] = [line.strip().lstrip('- ').lstrip('* ') for line in fr_text.split('\n') if line.strip()]

        return metadata


class PathAgent(BaseAgent):
    """路径规划智能体 - 规划个性化学习路径"""
    
    def __init__(self):
        super().__init__(
            name="path_agent",
            role="学习路径规划师",
            system_prompt="""你是一位专业的学习路径规划师。你需要根据学生的学习画像，规划科学、动态的个性化学习路径。

路径规划原则：
1. 基于学生当前知识水平
2. 符合认知规律（由浅入深）
3. 结合学生学习偏好
4. 设置合理的学习节奏
5. 包含阶段性目标
6. 支持动态调整"""
        )
    
    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})
        generated_resources = state.get("generated_resources", [])
        
        subject = task_params.get("subject", "")
        topic = task_params.get("topic", "")
        available_resources = task_params.get("available_resources", [])
        
        # 合并生成的资源和已有资源
        all_resources = []
        for r in generated_resources:
            all_resources.append({
                "id": r.get("title", ""),
                "type": r.get("type", ""),
                "title": r.get("title", ""),
                "difficulty": r.get("difficulty", "intermediate")
            })
        all_resources.extend(available_resources)
        
        prompt = f"""请为以下学生规划个性化学习路径：

学生画像：
{json.dumps(profile, ensure_ascii=False, indent=2)}

学习主题：{subject} - {topic}

可用学习资源：
{json.dumps(all_resources, ensure_ascii=False, indent=2)}

要求：
1. 设计循序渐进的学习步骤
2. 每个步骤指定学习资源和预估时间
3. 标注前置依赖关系
4. 设置阶段性检查点
5. 结合学生学习偏好和节奏
6. 返回JSON格式

输出格式：
{{
    "title": "学习路径标题",
    "subject": "科目",
    "topic": "主题",
    "estimated_total_hours": 总时长,
    "phases": [
        {{
            "phase": 1,
            "name": "阶段名称",
            "description": "阶段描述",
            "objectives": ["阶段目标"],
            "steps": [
                {{
                    "step": 1,
                    "action": "学习动作",
                    "resource_type": "资源类型",
                    "resource_title": "资源标题",
                    "estimated_minutes": 预估分钟,
                    "prerequisites": [],
                    "completion_criteria": "完成标准"
                }}
            ],
            "checkpoint": {{
                "type": "quiz/practice/project",
                "description": "检查点描述",
                "pass_criteria": "通过标准"
            }}
        }}
    ],
    "recommendations": ["学习建议"]
}}"""
        
        result = await self.llm.chat_json(self.build_messages(prompt), temperature=0.5)
        
        return {
            **state,
            "learning_path": result,
            "current_agent": self.name,
            "next_agents": []
        }


class TutorAgent(BaseAgent):
    """辅导答疑智能体 - 提供即时答疑服务"""
    
    def __init__(self):
        super().__init__(
            name="tutor_agent",
            role="智能辅导教师",
            system_prompt="""你是一位耐心、专业的智能辅导教师。你需要为学生提供即时的答疑解惑服务。

辅导原则：
1. 先理解学生的问题
2. 根据学生画像调整解释方式
3. 提供多角度、多模态的解答
4. 引导学生独立思考
5. 举一反三，拓展相关知识点

解答形式：
- 文字详解
- 图解说明（文字描述图示）
- 类比解释
- 分步骤讲解
- 相关例题"""
        )
    
    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})
        messages = state.get("messages", [])
        
        # 获取学生最近的问题
        question = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                question = msg.get("content", "")
                break
        
        if not question:
            question = task_params.get("question", "请帮我解答学习中的疑问")
        
        # 获取学生易错点
        error_patterns = profile.get("error_patterns", {})
        cognitive_style = profile.get("cognitive_style", {})
        
        prompt = f"""学生提问：{question}

学生画像信息：
- 认知风格：{json.dumps(cognitive_style, ensure_ascii=False)}
- 易错点：{json.dumps(error_patterns, ensure_ascii=False)}
- 知识基础：{json.dumps(profile.get('knowledge_base', {}), ensure_ascii=False)}

请提供全面的解答，包含：
1. 问题分析和核心概念解释
2. 分步骤详细解答
3. 图解说明（用文字描述图示）
4. 常见错误提醒
5. 相关知识点拓展
6. 类似例题（如有帮助）

返回JSON格式：
{{
    "question": "原问题",
    "core_concept": "核心概念",
    "detailed_answer": "详细解答（Markdown格式）",
    "visual_explanation": "图解说明（文字描述）",
    "step_by_step": ["步骤1", "步骤2", ...],
    "common_mistakes": ["易错点1", "易错点2"],
    "related_knowledge": ["相关知识点"],
    "practice_question": {{"question": "类似练习题", "answer": "答案", "hint": "提示"}},
    "learning_tips": ["学习建议"]
}}"""
        
        result = await self.llm.chat_json(self.build_messages(prompt), temperature=0.5)
        
        return {
            **state,
            "tutoring_response": result,
            "messages": [{
                "role": "assistant", 
                "content": json.dumps(result, ensure_ascii=False)
            }],
            "current_agent": self.name,
            "next_agents": []
        }


class EvaluationAgent(BaseAgent):
    """评估智能体 - 学习效果多维度评估"""
    
    def __init__(self):
        super().__init__(
            name="evaluation_agent",
            role="学习评估分析师",
            system_prompt="""你是一位专业的学习评估分析师。你需要对学生的学习效果进行多维度精准评估。

评估维度：
1. 知识掌握度：核心知识点的理解与记忆
2. 技能应用能力：知识在实际问题中的应用
3. 问题解决能力：复杂问题的分析与解决
4. 学习进度：与计划目标的对比
5. 学习投入度：学习时长、频率、专注度
6. 进步趋势：与历史表现的对比"""
        )
    
    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        profile = state.get("student_profile", {})
        
        # 模拟学习数据（实际应从数据库获取）
        learning_data = task_params.get("learning_data", {
            "total_study_hours": 45,
            "completed_resources": 12,
            "exercise_scores": [75, 82, 78, 88, 85],
            "recent_activities": ["view_lecture", "complete_exercise", "ask_question"],
            "error_rate": 0.25,
            "engagement_score": 80
        })
        
        prompt = f"""请对以下学生的学习效果进行多维度评估：

学生画像
{json.dumps(profile, ensure_ascii=False, indent=2)}

学习数据：
{json.dumps(learning_data, ensure_ascii=False, indent=2)}

要求：
1. 对每个维度进行评分（0-100）和分析
2. 指出优势和不足
3. 给出具体改进建议
4. 建议调整学习策略
5. 返回JSON格式

输出格式：
{{
    "evaluation_date": "日期",
    "overall_score": 综合分,
    "dimensions": {{
        "knowledge_mastery": {{"score": 分, "analysis": "分析"}},
        "skill_application": {{"score": 分, "analysis": "分析"}},
        "problem_solving": {{"score": 分, "analysis": "分析"}},
        "learning_progress": {{"score": 分, "analysis": "分析"}},
        "engagement": {{"score": 分, "analysis": "分析"}},
        "improvement_trend": {{"score": 分, "analysis": "分析"}}
    }},
    "strengths": ["优势1", "优势2"],
    "weaknesses": ["不足1", "不足2"],
    "recommendations": [
        {{"area": "改进领域", "suggestion": "具体建议", "priority": "high/medium/low"}}
    ],
    "learning_strategy_adjustments": ["策略调整建议"],
    "next_focus": ["下一步重点"]
}}"""
        
        result = await self.llm.chat_json(self.build_messages(prompt), temperature=0.4)
        
        return {
            **state,
            "evaluation": result,
            "current_agent": self.name,
            "next_agents": []
        }


class OrchestratorAgent(BaseAgent):
    """协调智能体 - 协调多智能体协同工作"""
    
    def __init__(self):
        super().__init__(
            name="orchestrator",
            role="多智能体协调器",
            system_prompt="""你是多智能体系统的协调器。你负责分析任务需求，决定调用哪些智能体，以及协调它们的工作流程。

你的职责：
1. 分析用户请求，确定任务类型
2. 决定需要调用的智能体及其顺序
3. 协调智能体间的数据传递
4. 整合多智能体的输出结果
5. 确保任务完成的完整性"""
        )
    async def process(self, state: AgentState) -> AgentState:
        task_params = state.get("task_params", {})
        task_type = state.get("task_type", "")
        messages = state.get("messages", [])
        current_agent = state.get("current_agent", "") # 获取当前刚刚执行完的节点

        # 🚀 修复核心：如果当前节点不是协调器自己，说明某个子智能体（如 profile_agent）刚刚跑完
        # 既然它们已经跑完了，并且返回了结果，协调器就应该直接收尾结束，而不是再次派发！
        if current_agent != self.name and current_agent != "":
            return {
                **state,
                "current_agent": self.name,
                "next_agents": [],  # 清空队列，准备退出
                "is_complete": True  # 标记完成
            }

        # ---- 以下是首次进入协调器（current_agent == "orchestrator" 或为空）时的正常派发逻辑 ----
        if task_type == "full_resource_generation":
            return {
                **state,
                "current_agent": self.name,
                "next_agents": [
                    "content_agent",
                    "exercise_agent",
                    "mindmap_agent",
                    "reading_agent",
                    "code_agent",
                    "ppt_slides_agent"
                ],
                "is_complete": False
            }
        elif task_type == "profile_building":
            return {
                **state,
                "current_agent": self.name,
                "next_agents": ["profile_agent"],
                "is_complete": False
            }
        elif task_type == "path_planning":
            return {
                **state,
                "current_agent": self.name,
                "next_agents": ["path_agent"],
                "is_complete": False
            }
        elif task_type == "tutoring":
            return {
                **state,
                "current_agent": self.name,
                "next_agents": ["tutor_agent"],
                "is_complete": False
            }
        elif task_type == "evaluation":
            return {
                **state,
                "current_agent": self.name,
                "next_agents": ["evaluation_agent"],
                "is_complete": False
            }
        else:
            # 根据对话内容智能判断
            last_message = messages[-1]["content"] if messages else ""
            analysis_prompt = f"""分析以下用户请求，判断需要调用哪些智能体。

用户请求：{last_message}

可用智能体：
- profile_agent: 构建/更新学习画像
- content_agent: 生成课程讲解文档
- exercise_agent: 生成练习题（含自动批阅）
- mindmap_agent: 生成思维导图
- reading_agent: 生成拓展阅读
- ppt_slides_agent: 生成课件讲义文档
- code_agent: 生成代码实操案例
- path_agent: 规划学习路径
- tutor_agent: 辅导答疑
- evaluation_agent: 学习评估

请返回需要调用的智能体列表（按优先级排序）：
{{"agents": ["agent1", "agent2"], "reasoning": "判断理由"}}"""
            
            result = await self.llm.chat_json(self.build_messages(analysis_prompt))
            agents = result.get("agents", ["tutor_agent"])
            
            return {
                **state,
                "current_agent": self.name,
                "next_agents": agents,
                "is_complete": False
            }

# ==================== 多智能体工作流 ====================

class MultiAgentSystem:
    """多智能体协同工作系统"""
    
    def __init__(self):
        # 初始化所有智能体
        self.agents = {
            "profile_agent": ProfileAgent(),
            "content_agent": ContentAgent(),
            "exercise_agent": ExerciseAgent(),
            "mindmap_agent": MindMapAgent(),
            "reading_agent": ReadingAgent(),
            "ppt_slides_agent": PPTSlidesAgent(),
            "code_agent": CodeAgent(),
            "path_agent": PathAgent(),
            "tutor_agent": TutorAgent(),
            "evaluation_agent": EvaluationAgent(),
            "orchestrator": OrchestratorAgent(),
        }
        
        # 构建LangGraph工作流
        self.workflow = self._build_workflow()
        self.checkpointer = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)
    
    def _build_workflow(self) -> StateGraph:
        """构建多智能体工作流图"""
        workflow = StateGraph(AgentState)
        
        # 添加所有智能体节点
        workflow.add_node("orchestrator", self._agent_node("orchestrator"))
        workflow.add_node("profile_agent", self._agent_node("profile_agent"))
        workflow.add_node("content_agent", self._agent_node("content_agent"))
        workflow.add_node("exercise_agent", self._agent_node("exercise_agent"))
        workflow.add_node("mindmap_agent", self._agent_node("mindmap_agent"))
        workflow.add_node("reading_agent", self._agent_node("reading_agent"))
        workflow.add_node("ppt_slides_agent", self._agent_node("ppt_slides_agent"))
        workflow.add_node("code_agent", self._agent_node("code_agent"))
        workflow.add_node("path_agent", self._agent_node("path_agent"))
        workflow.add_node("tutor_agent", self._agent_node("tutor_agent"))
        workflow.add_node("evaluation_agent", self._agent_node("evaluation_agent"))
        
        # 设置入口
        workflow.set_entry_point("orchestrator")
        
        # 条件路由：根据next_agents决定下一步
        # 建立路由映射，把各 agent 的名字映射到对应的节点
        routing_map = {name: name for name in self.agents.keys()}
        
        # 🚀 修复核心：显式将 "__end__" 映射到 LangGraph 的系统终点 END
        routing_map["__end__"] = END  
        
        # 或者如果你用字符串，也可以写成: routing_map[END] = END

        # 添加条件边
        workflow.add_conditional_edges(
            "orchestrator",
            self._route_next,
            routing_map
        )
        
        
        # 每个智能体处理完后返回协调器
        for agent_name in self.agents.keys():
            if agent_name != "orchestrator":
                workflow.add_edge(agent_name, "orchestrator")
        
        return workflow
    
    def _agent_node(self, agent_name: str):
        """创建智能体节点函数"""
        agent = self.agents[agent_name]
        
        async def node_func(state: AgentState) -> AgentState:
            return await agent.process(state)
        
        return node_func
    
    def _route_next(self, state: AgentState) -> str:
        """根据当前状态决定下一个执行的智能体"""
        next_agents = state.get("next_agents", [])
        if not next_agents:
            return END  # 或者返回 "__end__"
        return next_agents[0]
    
    async def run(self, task_type: str, task_params: Dict[str, Any], 
                  student_profile: Dict[str, Any] = None,
                  messages: List[Dict[str, str]] = None,
                  student_id: str = None) -> AgentState:
        """运行多智能体系统"""
        
        initial_state: AgentState = {
            "student_id": student_id,
            "student_profile": student_profile or {},
            "messages": messages or [],
            "conversation_context": {},
            "task_type": task_type,
            "task_params": task_params,
            "generated_resources": [],
            "extracted_features": {},
            "learning_path": None,
            "evaluation": None,
            "tutoring_response": None,
            "current_agent": "orchestrator",
            "next_agents": [],
            "is_complete": False,
            "error": None,
        }
        
        # 配置
        config = {"configurable": {"thread_id": student_id or "default"}}
        
        # 运行工作流
        result = await self.app.ainvoke(initial_state, config)
        
        return result
    
    async def generate_all_resources(self, subject: str, topic: str,
                                      profile: Dict[str, Any],
                                      student_id: str = None) -> Dict[str, Any]:
        """一键并行生成所有类型的学习资源"""

        task_params = {
            "subject": subject,
            "topic": topic,
            "action": "generate_all"
        }

        resource_agents = [
            ("content_agent", {"subject": subject, "topic": topic, "difficulty": "intermediate"}),
            ("exercise_agent", {"subject": subject, "topic": topic, "exercise_type": "mixed", "count": 10}),
            ("mindmap_agent", {"subject": subject, "topic": topic}),
            ("reading_agent", {"subject": subject, "topic": topic}),
            ("code_agent", {"subject": subject, "topic": topic, "language": "python"}),
            ("ppt_slides_agent", {"subject": subject, "topic": topic, "difficulty": "intermediate"}),
        ]

        state = {
            "student_id": student_id,
            "student_profile": profile,
            "messages": [],
            "conversation_context": {},
            "task_type": "full_resource_generation",
            "task_params": task_params,
            "generated_resources": [],
            "extracted_features": {},
            "learning_path": None,
            "evaluation": None,
            "tutoring_response": None,
            "current_agent": "",
            "next_agents": [],
            "is_complete": False,
            "error": None,
        }

        async def run_agent(agent_name: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
            """并行执行单个智能体"""
            agent = self.agents[agent_name]
            agent_state = {**state, "task_params": params}
            try:
                print(f"[资源生成] 开始调用 {agent_name} ...")
                result_state = await agent.process(agent_state)
                resources = result_state.get("generated_resources", [])
                print(f"[资源生成] {agent_name} 生成完成，产出 {len(resources)} 个资源")
                for r in resources:
                    print(f"  - 类型: {r.get('type', 'unknown')}, 标题: {r.get('title', '无标题')[:40]}")
                return resources
            except Exception as e:
                print(f"[警告] 智能体 {agent_name} 生成失败: {e}")
                import traceback
                traceback.print_exc()
                return []

        # 真正并行执行所有资源生成智能体
        results = await asyncio.gather(*[
            run_agent(agent_name, params)
            for agent_name, params in resource_agents
        ])

        # 汇总所有生成结果
        all_resources = []
        for resources in results:
            all_resources.extend(resources)

        print(f"[资源生成] 总共生成 {len(all_resources)} 个资源")

        # 如果某些类型缺失，生成默认资源补齐
        existing_types = {r.get("type") for r in all_resources}
        expected_types = {"lecture_doc", "exercise", "mind_map", "reading_material", "code_practice", "ppt_slides"}
        missing_types = expected_types - existing_types

        for missing_type in missing_types:
            print(f"[资源生成] 类型 {missing_type} 缺失，生成默认资源补齐")
            default_resource = self._create_default_resource(missing_type, subject, topic)
            all_resources.append(default_resource)

        # 生成学习路径（依赖资源生成结果，必须串行）
        path_params = {
            "subject": subject,
            "topic": topic,
            "available_resources": [
                {"id": r.get("title", ""), "type": r.get("type", ""),
                 "title": r.get("title", ""), "difficulty": r.get("difficulty", "intermediate")}
                for r in all_resources
            ]
        }
        path_state = {**state, "task_params": path_params, "generated_resources": all_resources}
        path_result = await self.agents["path_agent"].process(path_state)

        return {
            "resources": all_resources,
            "learning_path": path_result.get("learning_path"),
            "profile": profile
        }

    def _create_default_resource(self, resource_type: str, subject: str, topic: str) -> Dict[str, Any]:
        """创建默认资源，用于补齐缺失的类型"""
        type_titles = {
            "lecture_doc": f"{subject} - {topic} 课程讲解",
            "exercise": f"{subject} - {topic} 练习题",
            "mind_map": f"{subject} - {topic} 思维导图",
            "reading_material": f"{subject} - {topic} 拓展阅读",
            "code_practice": f"{subject} - {topic} 代码实操",
            "ppt_slides": f"{subject} - {topic} 课件讲义",
        }
        type_contents = {
            "lecture_doc": f"# {subject} - {topic}\n\n## 概述\n\n本节将介绍{topic}的核心概念和基本原理。\n\n## 学习目标\n\n1. 理解{topic}的基本概念\n2. 掌握{topic}的核心方法\n3. 能够运用{topic}解决实际问题\n\n## 详细内容\n\n待生成...",
            "exercise": json.dumps({
                "title": f"{subject} - {topic} 练习题",
                "subject": subject,
                "topic": topic,
                "total_questions": 5,
                "difficulty": "intermediate",
                "questions": [
                    {
                        "id": 1,
                        "type": "选择",
                        "question": f"关于{topic}，以下说法正确的是？",
                        "options": ["A. 选项A", "B. 选项B", "C. 选项C", "D. 选项D"],
                        "answer": "A",
                        "analysis": "这是解析",
                        "difficulty": "medium",
                        "knowledge_point": topic,
                        "score": 10
                    }
                ],
                "auto_gradable_count": 1,
                "total_score": 10
            }, ensure_ascii=False),
            "mind_map": json.dumps({
                "title": f"{subject} - {topic} 思维导图",
                "subject": subject,
                "topic": topic,
                "root": {
                    "name": topic,
                    "children": [
                        {"name": "核心概念", "description": f"{topic}的基本定义", "importance": "high"},
                        {"name": "应用场景", "description": f"{topic}的实际应用", "importance": "medium"},
                        {"name": "相关技术", "description": f"与{topic}相关的技术", "importance": "medium"}
                    ]
                }
            }, ensure_ascii=False),
            "reading_material": json.dumps({
                "title": f"{subject} - {topic} 拓展阅读",
                "subject": subject,
                "topic": topic,
                "materials": [
                    {
                        "id": 1,
                        "title": f"{topic}入门指南",
                        "type": "博客",
                        "source": "技术博客",
                        "summary": f"介绍{topic}的基础知识和入门方法",
                        "recommendation_reason": "适合初学者",
                        "difficulty_level": "beginner",
                        "estimated_reading_time": "15分钟",
                        "key_takeaways": ["基础概念", "入门方法"]
                    }
                ]
            }, ensure_ascii=False),
            "code_practice": json.dumps({
                "title": f"{subject} - {topic} 代码实操",
                "subject": subject,
                "topic": topic,
                "language": "python",
                "difficulty": "intermediate",
                "objectives": [f"掌握{topic}的编程实现"],
                "prerequisites": ["Python基础"],
                "steps": [
                    {
                        "step": 1,
                        "title": "基础实现",
                        "description": f"实现{topic}的基本功能",
                        "code": f"# {topic} 基础实现\nprint('Hello {topic}')",
                        "explanation": "这是基础代码示例",
                        "expected_output": "Hello {topic}",
                        "common_mistakes": ["语法错误"]
                    }
                ],
                "extended_exercises": [],
                "summary": f"通过本练习掌握{topic}的编程实现"
            }, ensure_ascii=False),
            "ppt_slides": f"# {subject} - {topic} 课件讲义\n\n> 科目：{subject} | 主题：{topic}\n\n---\n\n## 1. 课程导入\n\n本节课程将带领大家深入了解{topic}。\n\n## 2. 核心概念\n\n### 2.1 基本概念\n\n{topic}是{subject}中的重要内容...\n\n### 2.2 关键原理\n\n...\n\n## 3. 案例分析\n\n通过实际案例加深理解...\n\n## 4. 总结\n\n本节重点回顾...",
        }

        return {
            "type": resource_type,
            "title": type_titles.get(resource_type, f"{subject} - {topic}"),
            "subject": subject,
            "topic": topic,
            "difficulty": "intermediate",
            "content": type_contents.get(resource_type, ""),
            "generated_by": "default_agent",
            "format": "json" if resource_type in ["exercise", "mind_map", "reading_material", "code_practice"] else "markdown",
            "metadata": json.loads(type_contents.get(resource_type, "{}")) if resource_type in ["exercise", "mind_map", "reading_material", "code_practice"] else None
        }


# 全局多智能体系统实例
multi_agent_system = MultiAgentSystem()
