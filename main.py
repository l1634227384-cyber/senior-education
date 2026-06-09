"""
高等教育个性化学习资源智能体系统 - FastAPI 主应用
"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from agents import multi_agent_system

from models import (
    init_db, get_session_direct, async_session, Student, StudentProfile, LearningResource,
    ResourceFeedback, LearningPath, EvaluationResult, Conversation
)
# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description=settings.PROJECT_DESCRIPTION
)

# 模板和静态文件
templates = Jinja2Templates(directory=str(settings.TEMPLATE_DIR))
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")


# ==================== 数据库依赖 ====================

async def get_db() -> AsyncSession:
    """获取数据库会话依赖"""
    async with async_session() as session:
        yield session

# ==================== 请求/响应模型 ====================

class StudentCreateRequest(BaseModel):
    """创建学生请求"""
    name: str
    student_id: str
    major: Optional[str] = None
    grade: Optional[str] = None
    university: Optional[str] = None


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""
    student_id: str
    message: str
    conversation_type: str = "general"  # profile_building, tutoring, general


class ResourceGenerateRequest(BaseModel):
    """资源生成请求"""
    student_id: str
    subject: str
    topic: str
    resource_types: Optional[List[str]] = None  # 指定资源类型，None表示全部


class PathPlanRequest(BaseModel):
    """路径规划请求"""
    student_id: str
    subject: str
    topic: str


class EvaluationRequest(BaseModel):
    """评估请求"""
    student_id: str
    subject: Optional[str] = None
    learning_data: Optional[Dict[str, Any]] = None


class ResourceFeedbackRequest(BaseModel):
    """资源反馈请求"""
    student_id: str
    resource_id: str
    rating: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = None


class ExerciseGradeRequest(BaseModel):
    """习题批阅请求"""
    resource_id: str
    answers: Dict[str, str]  # {question_id: student_answer}


class ChatIntentRequest(BaseModel):
    """对话意图识别请求"""
    student_id: str
    message: str


# ==================== 工具函数 ====================

async def _get_student(db: AsyncSession, student_id: str) -> Student:
    """获取学生对象，不存在则抛 404"""
    result = await db.execute(
        select(Student).where(Student.student_id == student_id)
    )
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=404, detail="学生不存在")
    return student


async def _get_or_create_profile(db: AsyncSession, student_pk: int) -> StudentProfile:
    """获取或创建学生画像"""
    result = await db.execute(
        select(StudentProfile).where(StudentProfile.student_id == student_pk)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        profile = StudentProfile(
            student_id=student_pk,
            knowledge_base={},
            cognitive_style={"type": "unknown"},
            learning_preference={"preferred_formats": ["text"], "preferred_difficulty": "intermediate"},
            error_patterns={"常见错误类型": [], "易错知识点": []},
            learning_pace={"speed": "moderate", "avg_session_minutes": 30, "sessions_per_week": 3},
            goal_orientation={"short_term": [], "long_term": [], "priority": "balanced"},
            subject_interests={"primary": [], "secondary": [], "interest_level": 50},
            learning_habits={"best_time": "flexible", "note_taking": "digital"},
            profile_summary="新学生，画像待构建",
            profile_version=1
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
    return profile


def profile_to_dict(profile: StudentProfile) -> Dict[str, Any]:
    """将 ORM 画像对象转为字典"""
    return {
        "knowledge_base": profile.knowledge_base or {},
        "cognitive_style": profile.cognitive_style or {},
        "learning_preference": profile.learning_preference or {},
        "error_patterns": profile.error_patterns or {},
        "learning_pace": profile.learning_pace or {},
        "goal_orientation": profile.goal_orientation or {},
        "subject_interests": profile.subject_interests or {},
        "learning_habits": profile.learning_habits or {},
        "profile_summary": profile.profile_summary or "",
        "profile_version": profile.profile_version or 1
    }


def student_to_dict(student: Student) -> Dict[str, Any]:
    """将 ORM 学生对象转为字典"""
    return {
        "student_id": student.student_id,
        "name": student.name,
        "major": student.major,
        "grade": student.grade,
        "university": student.university,
        "created_at": student.created_at.isoformat() if student.created_at else None
    }


# ==================== API路由 ====================

@app.on_event("startup")
async def startup():
    """应用启动"""
    await init_db()
    print(f"{settings.PROJECT_NAME} v{settings.PROJECT_VERSION} 启动成功")
    print(f"访问地址: https://localhost:8000")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页"""
    return templates.TemplateResponse(request=request, name="index.html")


# ==================== 学生信息 ====================
@app.post("/api/students/register")
async def register_student(request: StudentCreateRequest, db: AsyncSession = Depends(get_db)):
    """注册新学生（学号必须不存在）"""
    existing = await db.execute(
        select(Student).where(Student.student_id == request.student_id)
    )
    student = existing.scalar_one_or_none()
    if student:
        raise HTTPException(status_code=400, detail="该学号已注册，请直接登录")

    student = Student(
        student_id=request.student_id,
        name=request.name,
        major=request.major,
        grade=request.grade,
        university=request.university
    )
    db.add(student)
    await db.commit()
    await db.refresh(student)

    # 初始化画像
    await _get_or_create_profile(db, student.id)

    return {"status": "success", "student": student_to_dict(student)}


class StudentLoginRequest(BaseModel):
    """登录请求"""
    student_id: str
    name: str


@app.post("/api/students/login")
async def login_student(request: StudentLoginRequest, db: AsyncSession = Depends(get_db)):
    """学生登录（验证学号和姓名）"""
    result = await db.execute(
        select(Student).where(Student.student_id == request.student_id)
    )
    student = result.scalar_one_or_none()
    if not student:
        raise HTTPException(status_code=404, detail="学号不存在，请先注册")
    if student.name != request.name:
        raise HTTPException(status_code=403, detail="姓名与学号不匹配")

    return {"status": "success", "student": student_to_dict(student)}


# 保留旧接口兼容
@app.post("/api/students")
async def create_student(request: StudentCreateRequest, db: AsyncSession = Depends(get_db)):
    """创建学生（兼容旧接口）"""
    return await register_student(request, db)

@app.get("/api/students/{student_id}")
async def get_student(student_id: str, db: AsyncSession = Depends(get_db)):
    """获取学生信息"""
    student = await _get_student(db, student_id)
    return student_to_dict(student)



@app.get("/api/students/{student_id}/conversations")
async def get_conversations(student_id: str, db: AsyncSession = Depends(get_db)):
    """获取学生对话历史"""
    student_result = await db.execute(
        select(Student).where(Student.student_id == student_id)
    )
    student = student_result.scalar_one_or_none()
    if not student:
        return {"messages": []}

    conv_result = await db.execute(
        select(Conversation).where(
            Conversation.student_id == student.id
        ).order_by(Conversation.created_at.desc()).limit(1)
    )
    conv = conv_result.scalar_one_or_none()
    messages = conv.messages if conv and conv.messages else []
    return {"messages": messages}


# ==================== 对话会话管理 API ====================

@app.get("/api/students/{student_id}/conversation-list")
async def list_conversations(student_id: str, db: AsyncSession = Depends(get_db)):
    """获取学生的所有对话会话列表（用于侧边栏）"""
    student = await _get_student(db, student_id)
    
    result = await db.execute(
        select(Conversation).where(
            Conversation.student_id == student.id
        ).order_by(Conversation.updated_at.desc())
    )
    conversations = result.scalars().all()
    
    conv_list = []
    for conv in conversations:
        # 取第一条用户消息作为预览
        preview = ""
        if conv.messages:
            for m in conv.messages:
                if m.get("role") == "user":
                    preview = m.get("content", "")[:50]
                    break
        conv_list.append({
            "id": conv.id,
            "session_id": conv.session_id,
            "title": conv.title or "新对话",
            "conversation_type": conv.conversation_type,
            "preview": preview,
            "is_active": conv.is_active or False,
            "message_count": len(conv.messages) if conv.messages else 0,
            "created_at": conv.created_at.isoformat() if conv.created_at else None,
            "updated_at": conv.updated_at.isoformat() if conv.updated_at else None
        })
    
    return {"conversations": conv_list, "total": len(conv_list)}


@app.post("/api/students/{student_id}/conversation-new")
async def create_conversation(student_id: str, db: AsyncSession = Depends(get_db)):
    """创建新对话会话"""
    student = await _get_student(db, student_id)
    
    # 将所有旧对话设为非活跃
    old_result = await db.execute(
        select(Conversation).where(
            Conversation.student_id == student.id,
            Conversation.is_active == True
        )
    )
    for old_conv in old_result.scalars().all():
        old_conv.is_active = False
    
    # 创建新对话
    new_conv = Conversation(
        student_id=student.id,
        session_id=str(uuid.uuid4()),
        conversation_type="general",
        title="新对话",
        messages=[],
        is_active=True
    )
    db.add(new_conv)
    await db.commit()
    await db.refresh(new_conv)
    
    return {
        "status": "success",
        "conversation": {
            "id": new_conv.id,
            "session_id": new_conv.session_id,
            "title": new_conv.title,
            "conversation_type": new_conv.conversation_type,
            "is_active": True,
            "message_count": 0,
            "created_at": new_conv.created_at.isoformat() if new_conv.created_at else None,
            "updated_at": new_conv.updated_at.isoformat() if new_conv.updated_at else None
        }
    }


@app.get("/api/students/{student_id}/conversation/{conv_id}")
async def load_conversation(student_id: str, conv_id: int, db: AsyncSession = Depends(get_db)):
    """加载指定对话会话的完整消息"""
    student = await _get_student(db, student_id)
    
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.student_id == student.id
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    # 将该对话设为活跃，其他设为非活跃
    old_result = await db.execute(
        select(Conversation).where(
            Conversation.student_id == student.id,
            Conversation.is_active == True,
            Conversation.id != conv_id
        )
    )
    for old_conv in old_result.scalars().all():
        old_conv.is_active = False
    
    conv.is_active = True
    await db.commit()
    
    return {
        "id": conv.id,
        "session_id": conv.session_id,
        "title": conv.title or "新对话",
        "conversation_type": conv.conversation_type,
        "messages": conv.messages or [],
        "is_active": True,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None
    }


@app.delete("/api/students/{student_id}/conversation/{conv_id}")
async def delete_conversation(student_id: str, conv_id: int, db: AsyncSession = Depends(get_db)):
    """删除指定对话会话"""
    student = await _get_student(db, student_id)
    
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conv_id,
            Conversation.student_id == student.id
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="对话不存在")
    
    was_active = conv.is_active
    await db.delete(conv)
    
    # 如果删除的是活跃对话，将最近的一个设为活跃
    if was_active:
        recent_result = await db.execute(
            select(Conversation).where(
                Conversation.student_id == student.id
            ).order_by(Conversation.updated_at.desc()).limit(1)
        )
        recent = recent_result.scalar_one_or_none()
        if recent:
            recent.is_active = True
    
    await db.commit()
    
    return {"status": "success", "message": "对话已删除"}

@app.get("/api/students/{student_id}/profile")
async def get_student_profile(student_id: str, db: AsyncSession = Depends(get_db)):
    """获取学生画像"""
    student = await _get_student(db, student_id)
    profile = await _get_or_create_profile(db, student.id)
    return profile_to_dict(profile)


# ==================== 对话式画像构建 ====================

@app.post("/api/chat/profile")
async def chat_profile_building(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """对话式画像构建"""
    student = await _get_student(db, request.student_id)
    profile = await _get_or_create_profile(db, student.id)
    profile_dict = profile_to_dict(profile)

    # 获取最近对话
    conv_result = await db.execute(
        select(Conversation).where(
            Conversation.student_id == student.id,
            Conversation.conversation_type == "profile_building"
        ).order_by(Conversation.created_at.desc()).limit(1)
    )
    conv = conv_result.scalar_one_or_none()

    messages = []
    if conv and conv.messages:
        messages = conv.messages[-10:]

    # 保存用户消息
    messages.append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    })

    # 调用多智能体系统
    result = await multi_agent_system.run(
        task_type="profile_building",
        task_params={"action": "build"},
        student_profile=profile_dict,
        messages=messages,
        student_id=request.student_id
    )

    # 更新画像到数据库
    if result.get("student_profile"):
        new_data = result["student_profile"]
        profile.knowledge_base = new_data.get("knowledge_base", profile.knowledge_base)
        profile.cognitive_style = new_data.get("cognitive_style", profile.cognitive_style)
        profile.learning_preference = new_data.get("learning_preference", profile.learning_preference)
        profile.error_patterns = new_data.get("error_patterns", profile.error_patterns)
        profile.learning_pace = new_data.get("learning_pace", profile.learning_pace)
        profile.goal_orientation = new_data.get("goal_orientation", profile.goal_orientation)
        profile.subject_interests = new_data.get("subject_interests", profile.subject_interests)
        profile.learning_habits = new_data.get("learning_habits", profile.learning_habits)
        profile.profile_summary = new_data.get("profile_summary", profile.profile_summary)
        profile.profile_version = (profile.profile_version or 1) + 1
        profile.updated_at = datetime.now()

    # 构建 AI 响应
    extracted = result.get("extracted_features", {})
    if extracted:
        summary = extracted.get("profile_summary", "")
        ai_response = f"""我已经分析了你的学习情况，以下是你的学习画像总结：

{summary}

📊 **已识别的维度：**
- 知识基础：{json.dumps(extracted.get('knowledge_base', {}), ensure_ascii=False)}
- 认知风格：{json.dumps(extracted.get('cognitive_style', {}), ensure_ascii=False)}
- 学习偏好：{json.dumps(extracted.get('learning_preference', {}), ensure_ascii=False)}
- 易错点：{json.dumps(extracted.get('error_patterns', {}), ensure_ascii=False)}

你可以继续和我聊天，我会不断完善你的学习画像。你也可以直接开始学习，告诉我你想学习什么内容！"""
    else:
        ai_response = "我已经收到你的信息，请继续告诉我更多关于你学习情况的细节，比如你的专业、感兴趣的领域、学习目标等。"

    # 保存 AI 回复到对话记录
    messages.append({
        "role": "assistant",
        "content": ai_response,
        "timestamp": datetime.now().isoformat()
    })

    # 更新或创建对话记录
    if conv:
        conv.messages = messages
        conv.updated_at = datetime.now()
        # 自动更新标题
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs and (not conv.title or conv.title == "新对话"):
            conv.title = user_msgs[0].get("content", "")[:30] + ("..." if len(user_msgs[0].get("content", "")) > 30 else "")
    else:
        # 自动生成标题
        first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
        title = first_user_msg.get("content", "")[:30] if first_user_msg else "新对话"
        if title != "新对话":
            title = title + ("..." if len(first_user_msg.get("content", "")) > 30 else "")
        
        conv = Conversation(
            student_id=student.id,
            session_id=str(uuid.uuid4()),
            conversation_type="profile_building",
            title=title,
            messages=messages
        )
        db.add(conv)

    await db.commit()

    return {
        "status": "success",
        "response": ai_response,
        "profile_updated": bool(extracted),
        "profile_summary": extracted.get("profile_summary", "")
    }


# ==================== 资源生成 ====================

@app.post("/api/resources/generate")
async def generate_resources(request: ResourceGenerateRequest, db: AsyncSession = Depends(get_db)):
    """生成个性化学习资源"""
    student = await _get_student(db, request.student_id)
    profile = await _get_or_create_profile(db, student.id)
    profile_dict = profile_to_dict(profile)

    # 调用多智能体系统生成资源
    result = await multi_agent_system.generate_all_resources(
        subject=request.subject,
        topic=request.topic,
        profile=profile_dict,
        student_id=request.student_id
    )

    resources = result.get("resources", [])
    learning_path_data = result.get("learning_path")

    print(f"[API] 收到 {len(resources)} 个待保存资源")
    for r in resources:
        print(f"  - 类型: {r.get('type', 'unknown')}, 标题: {r.get('title', '无标题')[:40]}")

    # 保存资源到数据库
    saved_resources = []
    for resource in resources:
        try:
            db_resource = LearningResource(
                resource_type=resource.get("type", ""),
                title=resource.get("title", ""),
                subject=resource.get("subject", ""),
                topic=resource.get("topic", ""),
                difficulty=resource.get("difficulty", "intermediate"),
                content=resource.get("content", ""),
                generated_by=resource.get("generated_by", ""),
                target_student_id=student.id,
                extra_data=resource.get("metadata", {}),
                generation_params={"subject": request.subject, "topic": request.topic}
            )
            db.add(db_resource)
            await db.flush()  # 获取真实数据库 ID
            saved_resources.append({
                "id": db_resource.id,
                "type": db_resource.resource_type,
                "title": db_resource.title,
                "subject": db_resource.subject,
                "topic": db_resource.topic,
                "difficulty": db_resource.difficulty,
                "content": db_resource.content,
                "generated_by": db_resource.generated_by,
                "created_at": db_resource.created_at.isoformat() if db_resource.created_at else None
            })
            print(f"[API] 资源已保存: {db_resource.resource_type} - {db_resource.title[:40]}")
        except Exception as e:
            print(f"[API错误] 保存资源失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存学习路径
    if learning_path_data:
        db_path = LearningPath(
            student_id=student.id,
            subject=request.subject,
            title=learning_path_data.get("title", f"{request.subject} - {request.topic}"),
            steps=learning_path_data.get("phases", []),
            current_step=0,
            progress=0.0,
            is_active=True
        )
        db.add(db_path)

    await db.commit()

    return {
        "status": "success",
        "resources": saved_resources,
        "learning_path": learning_path_data,
        "total_resources": len(saved_resources)
    }


@app.get("/api/resources/{resource_id}")
async def get_resource(resource_id: str, db: AsyncSession = Depends(get_db)):
    """获取单个资源详情"""
    result = await db.execute(
        select(LearningResource).where(LearningResource.id == resource_id)
    )
    resource = result.scalar_one_or_none()
    if not resource:
        raise HTTPException(status_code=404, detail="资源不存在")

    # 增加浏览次数
    resource.view_count = (resource.view_count or 0) + 1
    await db.commit()

    return {
        "id": str(resource.id),
        "type": resource.resource_type,
        "title": resource.title,
        "subject": resource.subject,
        "topic": resource.topic,
        "difficulty": resource.difficulty,
        "content": resource.content,
        "format": "json" if (resource.extra_data and resource.resource_type not in ("ppt_slides", "lecture_doc")) else "markdown",
        "metadata": resource.extra_data,
        "generated_by": resource.generated_by,
        "view_count": resource.view_count,
        "avg_rating": resource.avg_rating,
        "created_at": resource.created_at.isoformat() if resource.created_at else None
    }


@app.get("/api/students/{student_id}/resources")
async def get_student_resources(student_id: str, db: AsyncSession = Depends(get_db)):
    """获取学生的所有资源"""
    student = await _get_student(db, student_id)

    result = await db.execute(
        select(LearningResource).where(
            LearningResource.target_student_id == student.id
        ).order_by(LearningResource.created_at.desc())
    )
    resources = result.scalars().all()

    resource_list = [{
        "id": str(r.id),
        "type": r.resource_type,
        "title": r.title,
        "subject": r.subject,
        "topic": r.topic,
        "difficulty": r.difficulty,
        "generated_by": r.generated_by,
        "view_count": r.view_count,
        "avg_rating": r.avg_rating,
        "created_at": r.created_at.isoformat() if r.created_at else None
    } for r in resources]

    return {"resources": resource_list, "total": len(resource_list)}


@app.post("/api/resources/feedback")
async def submit_resource_feedback(request: ResourceFeedbackRequest, db: AsyncSession = Depends(get_db)):
    """提交资源反馈"""
    student = await _get_student(db, request.student_id)

    feedback = ResourceFeedback(
        student_id=student.id,
        resource_id=int(request.resource_id),
        rating=request.rating,
        feedback_text=request.feedback_text
    )
    db.add(feedback)

    # 更新资源平均评分
    avg_result = await db.execute(
        select(func.avg(ResourceFeedback.rating)).where(
            ResourceFeedback.resource_id == int(request.resource_id)
        )
    )
    avg_rating = avg_result.scalar()
    if avg_rating is not None:
        resource = await db.get(LearningResource, int(request.resource_id))
        if resource:
            resource.avg_rating = float(avg_rating)

    await db.commit()

    return {"status": "success", "message": "反馈已提交"}


@app.post("/api/resources/{resource_id}/grade")
async def grade_exercise(resource_id: str, request: ExerciseGradeRequest, db: AsyncSession = Depends(get_db)):
    """习题一键批阅"""
    result = await db.execute(
        select(LearningResource).where(LearningResource.id == resource_id)
    )
    resource = result.scalar_one_or_none()
    if not resource:
        raise HTTPException(status_code=404, detail="资源不存在")
    if resource.resource_type != "exercise":
        raise HTTPException(status_code=400, detail="该资源不是习题")

    metadata = resource.extra_data or {}
    if not metadata or "questions" not in metadata:
        raise HTTPException(status_code=400, detail="习题数据不完整")

    # 调用 ExerciseAgent 的批阅功能
    from agents import ExerciseAgent
    agent = ExerciseAgent()
    grade_result = await agent.grade_exercise(metadata, request.answers)

    return {
        "status": "success",
        "resource_id": resource_id,
        "grading_result": grade_result
    }


@app.post("/api/chat/intent")
async def detect_chat_intent(request: ChatIntentRequest, db: AsyncSession = Depends(get_db)):
    """检测对话意图，判断是否触发资源生成"""
    student = await _get_student(db, request.student_id)
    profile = await _get_or_create_profile(db, student.id)
    profile_dict = profile_to_dict(profile)

    message = request.message.strip()

    # 简单规则匹配 + LLM 意图识别
    import re

    # 1. 直接匹配资源生成请求
    gen_patterns = [
        r'(?:给我|我要|帮我|请给我|生成|创建|制作).{0,10}(?:学习资料|课件|讲义|练习题|思维导图|资料)',
        r'(?:学|学习|复习|预习).{0,5}(?:资料|资源|课件|讲义)',
        r'(?:需要|想要).{0,5}(?:资料|资源|课件|讲义|练习)',
    ]
    for pattern in gen_patterns:
        if re.search(pattern, message):
            # 尝试提取科目和主题
            subject_topic = _extract_subject_topic(message)
            if subject_topic:
                return {
                    "intent": "generate_resources",
                    "subject": subject_topic[0],
                    "topic": subject_topic[1],
                    "confidence": 0.9,
                    "message": "检测到学习资料生成请求"
                }

    # 2. 使用 LLM 进行意图识别
    intent_prompt = """分析以下学生消息，判断其意图：

学生消息：""" + message + """

请判断意图类型（返回JSON）：
{
    "intent": "generate_resources|profile_building|tutoring|general",
    "subject": "如果意图是生成资源，提取科目",
    "topic": "如果意图是生成资源，提取主题",
    "confidence": 0.0-1.0,
    "reasoning": "判断理由"
}

意图说明：
- generate_resources: 学生明确要求学习资料、课件、讲义、练习题等
- profile_building: 学生描述自己的学习情况、兴趣、目标等
- tutoring: 学生提问具体知识点问题
- general: 其他一般性对话"""

    from agents import LLMClient
    llm = LLMClient()
    intent_result = await llm.chat_json([
        {"role": "system", "content": "你是一个对话意图识别助手。"},
        {"role": "user", "content": intent_prompt}
    ], temperature=0.3)

    intent = intent_result.get("intent", "general")
    confidence = intent_result.get("confidence", 0.5)

    if intent == "generate_resources" and confidence >= 0.7:
        subject = intent_result.get("subject", "")
        topic = intent_result.get("topic", "")
        if not subject or not topic:
            subject_topic = _extract_subject_topic(message)
            if subject_topic:
                subject, topic = subject_topic
        return {
            "intent": "generate_resources",
            "subject": subject,
            "topic": topic,
            "confidence": confidence,
            "message": "AI识别到学习资料生成意图"
        }

    return {
        "intent": intent,
        "confidence": confidence,
        "message": intent_result.get("reasoning", "")
    }


def _extract_subject_topic(message: str) -> Optional[tuple]:
    """从消息中提取科目和主题"""
    import re

    # 常见模式："给我XXX的YYY资料" -> subject=XXX, topic=YYY
    patterns = [
        r'(?:给我|我要|帮我|生成|创建|制作)\s*(\S+?)\s*的\s*(\S+?)\s*(?:资料|资源|课件|讲义|练习|学习)',
        r'(?:学|学习|复习|预习)\s*(\S+?)\s*的\s*(\S+)',
        r'(\S+?)\s*的\s*(\S+?)\s*(?:资料|资源|课件|讲义|练习)',
        r'(?:生成|创建|制作)\s*(\S+?)\s+(\S+?)\s*(?:资料|资源|课件|讲义|练习)',
    ]

    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            return (match.group(1), match.group(2))

    # 尝试匹配 "给我回溯法的学习资料" 这种格式
    simple = re.search(r'(?:给我|我要|帮我|生成)\s*(\S+?)\s*(?:的)?\s*(?:学习资料|资料|课件|讲义)', message)
    if simple:
        topic = simple.group(1)
        # 尝试推断科目
        subject_map = {
            "回溯法": "算法设计与分析", "动态规划": "算法设计与分析", "贪心": "算法设计与分析",
            "二叉树": "数据结构", "链表": "数据结构", "图": "数据结构",
            "微积分": "高等数学", "线性代数": "高等数学", "概率论": "高等数学",
            "Python": "程序设计", "Java": "程序设计", "C++": "程序设计",
        }
        subject = subject_map.get(topic, "计算机科学")
        return (subject, topic)

    return None


# ==================== 学习路径 ====================

@app.post("/api/path/plan")
async def plan_learning_path(request: PathPlanRequest, db: AsyncSession = Depends(get_db)):
    """规划学习路径"""
    student = await _get_student(db, request.student_id)
    profile = await _get_or_create_profile(db, student.id)
    profile_dict = profile_to_dict(profile)

    # 获取已有资源
    result = await db.execute(
        select(LearningResource).where(LearningResource.target_student_id == student.id)
    )
    resources = result.scalars().all()
    available_resources = [
        {
            "id": str(r.id),
            "type": r.resource_type,
            "title": r.title,
            "difficulty": r.difficulty or "intermediate"
        }
        for r in resources
    ]

    agent_result = await multi_agent_system.run(
        task_type="path_planning",
        task_params={
            "subject": request.subject,
            "topic": request.topic,
            "available_resources": available_resources
        },
        student_profile=profile_dict,
        student_id=request.student_id
    )

    path = agent_result.get("learning_path")
    if path:
        # 将旧路径设为非活跃
        old_paths = await db.execute(
            select(LearningPath).where(
                LearningPath.student_id == student.id,
                LearningPath.is_active == True
            )
        )
        for old in old_paths.scalars().all():
            old.is_active = False

        db_path = LearningPath(
            student_id=student.id,
            subject=request.subject,
            title=path.get("title", f"{request.subject} - {request.topic}"),
            steps=path.get("phases", []),
            current_step=0,
            progress=0.0,
            is_active=True
        )
        db.add(db_path)
        await db.commit()

    return {
        "status": "success",
        "learning_path": path
    }


@app.get("/api/students/{student_id}/path")
async def get_learning_path(student_id: str, db: AsyncSession = Depends(get_db)):
    """获取学习路径"""
    student = await _get_student(db, student_id)

    result = await db.execute(
        select(LearningPath).where(
            LearningPath.student_id == student.id,
            LearningPath.is_active == True
        ).order_by(LearningPath.created_at.desc()).limit(1)
    )
    path = result.scalar_one_or_none()

    if not path:
        raise HTTPException(status_code=404, detail="学习路径不存在")

    return {
        "id": path.id,
        "data": path.steps,
        "subject": path.subject,
        "title": path.title,
        "progress": path.progress,
        "current_step": path.current_step,
        "created_at": path.created_at.isoformat() if path.created_at else None
    }


# ==================== 智能辅导 ====================

@app.post("/api/tutor/ask")
async def ask_tutor(request: ChatRequest, db: AsyncSession = Depends(get_db)):
    """智能辅导答疑"""
    student = await _get_student(db, request.student_id)
    profile = await _get_or_create_profile(db, student.id)
    profile_dict = profile_to_dict(profile)

    # 获取最近对话
    conv_result = await db.execute(
        select(Conversation).where(
            Conversation.student_id == student.id,
            Conversation.conversation_type == "tutoring"
        ).order_by(Conversation.created_at.desc()).limit(1)
    )
    conv = conv_result.scalar_one_or_none()
    messages = conv.messages[-6:] if conv and conv.messages else []

    # 添加用户消息
    messages.append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.now().isoformat()
    })

    result = await multi_agent_system.run(
        task_type="tutoring",
        task_params={"question": request.message},
        student_profile=profile_dict,
        messages=messages,
        student_id=request.student_id
    )

    response = result.get("tutoring_response", {})

    # 格式化回复
    if response:
        answer_text = f"""## 📝 问题解答

**核心概念：** {response.get('core_concept', '')}

{response.get('detailed_answer', '')}

### 🔍 图解说明
{response.get('visual_explanation', '')}

### 📋 分步讲解
{chr(10).join(['- ' + str(step) for step in response.get('step_by_step', [])])}

### ⚠️ 常见错误
{chr(10).join(['- ' + str(err) for err in response.get('common_mistakes', [])])}

### 📚 相关知识点
{chr(10).join(['- ' + str(k) for k in response.get('related_knowledge', [])])}

### 💡 学习建议
{chr(10).join(['- ' + str(tip) for tip in response.get('learning_tips', [])])}
"""
    else:
        answer_text = "抱歉，我暂时无法回答这个问题，请换个方式提问。"

    # 保存对话记录
    messages.append({
        "role": "assistant",
        "content": answer_text,
        "timestamp": datetime.now().isoformat()
    })

    if conv:
        conv.messages = messages
        conv.updated_at = datetime.now()
        # 自动更新标题
        user_msgs = [m for m in messages if m.get("role") == "user"]
        if user_msgs and (not conv.title or conv.title == "新对话"):
            conv.title = user_msgs[0].get("content", "")[:30] + ("..." if len(user_msgs[0].get("content", "")) > 30 else "")
    else:
        # 自动生成标题
        first_user_msg = next((m for m in messages if m.get("role") == "user"), None)
        title = first_user_msg.get("content", "")[:30] if first_user_msg else "新对话"
        if title != "新对话":
            title = title + ("..." if len(first_user_msg.get("content", "")) > 30 else "")
        
        conv = Conversation(
            student_id=student.id,
            session_id=str(uuid.uuid4()),
            conversation_type="tutoring",
            title=title,
            messages=messages
        )
        db.add(conv)

    await db.commit()

    return {
        "status": "success",
        "response": answer_text,
        "detailed": response
    }


# ==================== 学习评估 ====================

@app.post("/api/evaluation/assess")
async def evaluate_learning(request: EvaluationRequest, db: AsyncSession = Depends(get_db)):
    """学习效果评估"""
    student = await _get_student(db, request.student_id)
    profile = await _get_or_create_profile(db, student.id)
    profile_dict = profile_to_dict(profile)

    # 收集学习数据
    learning_data = request.learning_data
    if not learning_data:
        # 从已有数据计算
        res_result = await db.execute(
            select(LearningResource).where(LearningResource.target_student_id == student.id)
        )
        student_resources = res_result.scalars().all()

        fb_result = await db.execute(
            select(ResourceFeedback).where(ResourceFeedback.student_id == student.id)
        )
        feedbacks = fb_result.scalars().all()

        learning_data = {
            "total_resources_generated": len(student_resources),
            "feedback_count": len(feedbacks),
            "avg_rating": sum(f.rating for f in feedbacks) / len(feedbacks) if feedbacks else 0,
            "resource_types": list(set(r.resource_type for r in student_resources)),
            "profile_version": profile.profile_version or 1
        }

    result = await multi_agent_system.run(
        task_type="evaluation",
        task_params={"learning_data": learning_data},
        student_profile=profile_dict,
        student_id=request.student_id
    )

    evaluation = result.get("evaluation", {})

    # 保存评估结果
    db_eval = EvaluationResult(
        student_id=student.id,
        evaluation_type="comprehensive",
        subject=request.subject,
        dimensions=evaluation.get("dimensions", {}),
        strengths=evaluation.get("strengths", []),
        weaknesses=evaluation.get("weaknesses", []),
        recommendations=evaluation.get("recommendations", []),
        summary=evaluation.get("summary", "")
    )
    db.add(db_eval)
    await db.commit()

    return {
        "status": "success",
        "evaluation": evaluation
    }


@app.get("/api/students/{student_id}/evaluations")
async def get_evaluations(student_id: str, db: AsyncSession = Depends(get_db)):
    """获取评估历史"""
    student = await _get_student(db, student_id)

    result = await db.execute(
        select(EvaluationResult).where(
            EvaluationResult.student_id == student.id
        ).order_by(EvaluationResult.created_at.desc())
    )
    evaluations = result.scalars().all()

    eval_list = [{
        "data": {
            "overall_score": getattr(e, "overall_score", None),
            "dimensions": e.dimensions,
            "strengths": e.strengths,
            "weaknesses": e.weaknesses,
            "recommendations": e.recommendations,
            "summary": e.summary
        },
        "created_at": e.created_at.isoformat() if e.created_at else None
    } for e in evaluations]

    return {"evaluations": eval_list, "total": len(eval_list)}


# ==================== WebSocket 实时对话 ====================

@app.websocket("/ws/chat/{student_id}")
async def websocket_chat(websocket: WebSocket, student_id: str):
    """WebSocket实时对话"""
    await websocket.accept()

    # 使用独立数据库会话
    async with async_session() as db:
        # 确保学生存在
        result = await db.execute(
            select(Student).where(Student.student_id == student_id)
        )
        student = result.scalar_one_or_none()
        if not student:
            await websocket.send_json({"error": "学生不存在"})
            await websocket.close()
            return

        profile = await _get_or_create_profile(db, student.id)
        profile_dict = profile_to_dict(profile)

        # 获取当前活跃对话，如果没有则创建一个
        conv_result = await db.execute(
            select(Conversation).where(
                Conversation.student_id == student.id,
                Conversation.is_active == True
            ).order_by(Conversation.updated_at.desc()).limit(1)
        )
        conv = conv_result.scalar_one_or_none()
        
        if not conv:
            # 尝试获取最近一次对话
            conv_result = await db.execute(
                select(Conversation).where(
                    Conversation.student_id == student.id
                ).order_by(Conversation.updated_at.desc()).limit(1)
            )
            conv = conv_result.scalar_one_or_none()
        
        if not conv:
            # 创建新对话
            conv = Conversation(
                student_id=student.id,
                session_id=str(uuid.uuid4()),
                conversation_type="general",
                title="新对话",
                messages=[],
                is_active=True
            )
            db.add(conv)
            await db.commit()
            await db.refresh(conv)
        
        messages = conv.messages if conv and conv.messages else []

        try:
            while True:
                # 接收消息
                data = await websocket.receive_json()
                message = data.get("message", "")
                chat_type = data.get("type", "general")
                req_conversation_id = data.get("conversation_id")

                # ================= 加上日志打印 =================
                print(f"\n[WS 收到消息] 学生: {student_id}, 类型: {chat_type}, 内容: {message}")
                # ===============================================

                # 如果客户端指定了不同的对话ID，切换对话
                if req_conversation_id and req_conversation_id != (conv.id if conv else None):
                    new_conv_result = await db.execute(
                        select(Conversation).where(
                            Conversation.id == req_conversation_id,
                            Conversation.student_id == student.id
                        )
                    )
                    new_conv = new_conv_result.scalar_one_or_none()
                    if new_conv:
                        # 将旧对话设为非活跃
                        if conv:
                            conv.is_active = False
                        conv = new_conv
                        conv.is_active = True
                        await db.commit()
                        await db.refresh(conv)
                        messages = conv.messages if conv.messages else []

                # 自动更新对话标题（使用第一条用户消息的前30字）
                if conv and (not conv.title or conv.title == "新对话") and messages:
                    user_msgs = [m for m in messages if m.get("role") == "user"]
                    if not user_msgs:  # 这是第一条用户消息
                        conv.title = message[:30] + ("..." if len(message) > 30 else "")
                        conv.conversation_type = chat_type
                
                # 刷新 conv 的 messages 引用
                if conv:
                    await db.refresh(conv)
                    messages = conv.messages if conv.messages else []

                # 根据类型路由
                if chat_type == "profile":
                    messages.append({
                        "role": "user", "content": message,
                        "timestamp": datetime.now().isoformat()
                    })

                    print("[WS 智能体] 正在调用 multi_agent_system.run (画像构建)...")
                    try:
                        result = await asyncio.wait_for(
                            multi_agent_system.run(
                                task_type="profile_building",
                                task_params={"action": "build"},
                                student_profile=profile_dict,
                                messages=messages[-10:],
                                student_id=student_id
                            ), 
                        timeout=30.0
                        )
                        print(f"[WS 智能体] 调用结束 (画像构建)...")
                    except asyncio.TimeoutError:
                        print("[WS 错误] ❌ 智能体系统调用超时(30秒内未响应),已自动熔断!")
                        response = "抱歉，智能体助手响应超时，请稍后再试或检查网络。"
                        result = {}
                    print(f"[WS 智能体] 调用结束 (画像构建), 返回结果大小: {len(str(result))} 字符")

                    profile_updated = False
                    if result.get("student_profile"):
                        new_data = result["student_profile"]
                        profile.knowledge_base = new_data.get("knowledge_base", profile.knowledge_base)
                        profile.cognitive_style = new_data.get("cognitive_style", profile.cognitive_style)
                        profile.learning_preference = new_data.get("learning_preference", profile.learning_preference)
                        profile.error_patterns = new_data.get("error_patterns", profile.error_patterns)
                        profile.learning_pace = new_data.get("learning_pace", profile.learning_pace)
                        profile.goal_orientation = new_data.get("goal_orientation", profile.goal_orientation)
                        profile.subject_interests = new_data.get("subject_interests", profile.subject_interests)
                        profile.learning_habits = new_data.get("learning_habits", profile.learning_habits)
                        profile.profile_summary = new_data.get("profile_summary", profile.profile_summary)
                        profile.profile_version = (profile.profile_version or 1) + 1
                        profile.updated_at = datetime.now()
                        await db.commit()
                        profile_updated = True
                        profile_dict = profile_to_dict(profile)

                    extracted = result.get("extracted_features", {})
                    summary = extracted.get('profile_summary', '请继续告诉我更多信息。')
                    import re
                    summary = re.sub(r'<strong>(.*?)</strong>', r'**\1**', summary, flags=re.IGNORECASE)
                    summary = re.sub(r'<br\s*/?>', '\n', summary, flags=re.IGNORECASE)
                    summary = re.sub(r'<[^>]+>', '', summary)
                    response = f"已更新你的学习画像！{summary}"

                elif chat_type == "tutor":
                    messages.append({
                        "role": "user", "content": message,
                        "timestamp": datetime.now().isoformat()
                    })

                    print("[WS 智能体] 正在调用 multi_agent_system.run (智能辅导)...")
                    result = await multi_agent_system.run(
                        task_type="tutoring",
                        task_params={"question": message},
                        student_profile=profile_dict,
                        messages=messages[-6:],
                        student_id=student_id
                    )
                    print(f"[WS 智能体] 调用结束 (智能辅导), 返回结果大小: {len(str(result))} 字符")

                    tut_response = result.get("tutoring_response", {})
                    response = tut_response.get("detailed_answer", "请重新描述你的问题。")

                else:
                    messages.append({
                        "role": "user", "content": message,
                        "timestamp": datetime.now().isoformat()
                    })
                    print("[WS 智能体] 正在调用 multi_agent_system.run (通用对话)...")
                    result = await multi_agent_system.run(
                        task_type="tutoring",
                        task_params={"question": message},
                        student_profile=profile_dict,
                        messages=messages[-6:],
                        student_id=student_id
                    )
                    tut_response = result.get("tutoring_response", {})
                    response = tut_response.get("detailed_answer", "你好！我是你的智能学习助手。你可以：\n1. 和我聊聊你的学习情况（画像构建）\n2. 提问学习问题（智能辅导）\n3. 输入'生成 [科目] [主题] 的学习资料'来获取学习资源")

                messages.append({
                    "role": "assistant", "content": response,
                    "timestamp": datetime.now().isoformat()
                })

                # 更新对话标题（第一条用户消息）
                user_msgs = [m for m in messages if m.get("role") == "user"]
                if len(user_msgs) == 1 and (not conv.title or conv.title == "新对话"):
                    conv.title = user_msgs[0].get("content", "")[:30] + ("..." if len(user_msgs[0].get("content", "")) > 30 else "")

                # 保存对话
                print("[WS 数据库] 正在保存对话记录到数据库...")
                if conv:
                    conv.messages = messages
                    conv.updated_at = datetime.now()
                    conv.is_active = True
                await db.commit()
                print("[WS 数据库] 对话记录保存成功。")

                print("[WS 发送] 正在将结果推送到前端...")
                await websocket.send_json({
                    "type": chat_type,
                    "response": response,
                    "profile_updated": profile_updated if chat_type == "profile" and 'profile_updated' in locals() else False,
                    "conversation_id": conv.id if conv else None,
                    "conversation_title": conv.title if conv else "新对话",
                    "timestamp": datetime.now().isoformat()
                })
                print("[WS 流程完成] 等待下一次用户输入。\n")

        except WebSocketDisconnect:
            print(f"学生 {student_id} 断开WebSocket连接")


# ==================== 系统信息 ====================

@app.get("/api/system/info")
async def system_info(db: AsyncSession = Depends(get_db)):
    """系统信息"""
    students_count = (await db.execute(select(func.count(Student.id)))).scalar()
    resources_count = (await db.execute(select(func.count(LearningResource.id)))).scalar()

    return {
        "name": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "students_count": students_count or 0,
        "resources_count": resources_count or 0,
        "agent_types": settings.AGENT_TYPES,
        "resource_types": settings.RESOURCE_TYPES,
        "profile_dimensions": settings.PROFILE_DIMENSIONS
    }


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
