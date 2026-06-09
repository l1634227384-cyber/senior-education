"""
高等教育个性化学习资源智能体系统 - 数据模型定义
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from config import settings

Base = declarative_base()

# 异步引擎
engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# ==================== 学生相关模型 ====================

class Student(Base):
    """学生基础信息"""
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    major = Column(String(200))                    # 专业
    grade = Column(String(20))                     # 年级
    university = Column(String(200))               # 学校
    email = Column(String(200))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关联
    profile = relationship("StudentProfile", back_populates="student", uselist=False)
    learning_records = relationship("LearningRecord", back_populates="student")
    conversations = relationship("Conversation", back_populates="student")
    resource_feedbacks = relationship("ResourceFeedback", back_populates="student")


class StudentProfile(Base):
    """学生多维度画像 - 不少于6个维度"""
    __tablename__ = "student_profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"), unique=True)
    
    # 维度1: 知识基础 (0-100分)
    knowledge_base = Column(JSON, default=dict)  
    # 如: {"数学": 85, "编程": 60, "数据结构": 70, "算法": 55}
    
    # 维度2: 认知风格
    cognitive_style = Column(JSON, default=dict)
    # 如: {"type": "visual", "visual_score": 80, "verbal_score": 60, 
    #       "active_score": 70, "reflective_score": 65}
    
    # 维度3: 学习偏好
    learning_preference = Column(JSON, default=dict)
    # 如: {"preferred_formats": ["video", "text", "code"], 
    #       "preferred_duration": "30min", "preferred_difficulty": "medium"}
    
    # 维度4: 易错点偏好
    error_patterns = Column(JSON, default=dict)
    # 如: {"常见错误类型": ["概念混淆", "计算失误"], 
    #       "易错知识点": ["递归", "指针"], "错误频率": {...}}
    
    # 维度5: 学习节奏
    learning_pace = Column(JSON, default=dict)
    # 如: {"speed": "moderate", "avg_session_minutes": 45, 
    #       "sessions_per_week": 5, "completion_rate": 0.85}
    
    # 维度6: 目标导向
    goal_orientation = Column(JSON, default=dict)
    # 如: {"short_term": ["通过期末考试", "完成课程项目"], 
    #       "long_term": ["成为全栈工程师"], "priority": "practical_skills"}
    
    # 维度7: 学科兴趣
    subject_interests = Column(JSON, default=dict)
    # 如: {"primary": ["人工智能", "数据科学"], 
    #       "secondary": ["Web开发"], "interest_level": 85}
    
    # 维度8: 学习习惯
    learning_habits = Column(JSON, default=dict)
    # 如: {"best_time": "morning", "note_taking": "digital", 
    #       "review_frequency": "daily", "group_study": false}
    
    # 综合画像摘要（由LLM生成）
    profile_summary = Column(Text)
    
    # 画像版本控制（支持随学随新）
    profile_version = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关联
    student = relationship("Student", back_populates="profile")


# ==================== 对话相关模型 ====================

class Conversation(Base):
    """对话记录 - 用于画像构建和智能辅导"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    session_id = Column(String(100), index=True)
    conversation_type = Column(String(50))  # "profile_building", "tutoring", "general"
    messages = Column(JSON, default=list)   # [{"role": "user/assistant", "content": "...", "timestamp": "..."}]
    extracted_features = Column(JSON, default=dict)  # 从对话中提取的特征
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # 关联
    student = relationship("Student", back_populates="conversations")


# ==================== 学习资源模型 ====================

class LearningResource(Base):
    """学习资源"""
    __tablename__ = "learning_resources"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    resource_type = Column(String(50), nullable=False)  # lecture_doc, mind_map, exercise, etc.
    title = Column(String(300), nullable=False)
    description = Column(Text)
    subject = Column(String(200))
    topic = Column(String(200))
    difficulty = Column(String(20))  # beginner, intermediate, advanced
    content = Column(Text)           # 文本内容或JSON
    file_path = Column(String(500))  # 文件存储路径
    extra_data = Column(JSON, default=dict)  # 额外元数据（原名metadata，与SQLAlchemy保留字段冲突）
    tags = Column(JSON, default=list)
    
    # 生成信息
    generated_by = Column(String(100))  # 生成该资源的智能体
    target_student_id = Column(Integer, ForeignKey("students.id"), nullable=True)  # 个性化目标学生
    generation_params = Column(JSON, default=dict)  # 生成参数
    
    # 统计
    view_count = Column(Integer, default=0)
    use_count = Column(Integer, default=0)
    avg_rating = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class ResourceFeedback(Base):
    """资源反馈"""
    __tablename__ = "resource_feedbacks"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    resource_id = Column(Integer, ForeignKey("learning_resources.id"))
    rating = Column(Integer)  # 1-5
    feedback_text = Column(Text)
    helpfulness = Column(Float)  # 有用程度 0-1
    created_at = Column(DateTime, default=datetime.now)
    
    # 关联
    student = relationship("Student", back_populates="resource_feedbacks")


# ==================== 学习路径模型 ====================

class LearningPath(Base):
    """个性化学习路径"""
    __tablename__ = "learning_paths"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    subject = Column(String(200))
    title = Column(String(300))
    description = Column(Text)
    
    # 路径步骤 (有序)
    steps = Column(JSON, default=list)
    # [{"order": 1, "resource_id": 1, "type": "lecture_doc", 
    #   "estimated_time": 45, "prerequisites": [], "objectives": [...]}, ...]
    
    # 状态
    current_step = Column(Integer, default=0)
    progress = Column(Float, default=0.0)  # 0-100
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class LearningRecord(Base):
    """学习记录"""
    __tablename__ = "learning_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    resource_id = Column(Integer, ForeignKey("learning_resources.id"), nullable=True)
    activity_type = Column(String(50))  # view, complete, exercise_attempt, question, etc.
    duration_minutes = Column(Integer, default=0)
    score = Column(Float, nullable=True)  # 练习得分
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.now)
    
    # 关联
    student = relationship("Student", back_populates="learning_records")


# ==================== 评估模型 ====================

class EvaluationResult(Base):
    """学习效果评估结果"""
    __tablename__ = "evaluation_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    evaluation_type = Column(String(50))  # weekly, chapter, comprehensive
    subject = Column(String(200))
    
    # 多维度评估
    dimensions = Column(JSON, default=dict)
    # {"knowledge_mastery": 75, "skill_application": 68, 
    #  "problem_solving": 72, "progress_rate": 80, "engagement": 85}
    
    # 评估建议
    strengths = Column(JSON, default=list)
    weaknesses = Column(JSON, default=list)
    recommendations = Column(JSON, default=list)
    summary = Column(Text)
    
    created_at = Column(DateTime, default=datetime.now)


async def init_db():
    """初始化数据库"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session():
    """获取数据库会话（异步上下文管理器）"""
    async with async_session() as session:
        yield session


async def get_session_direct() -> AsyncSession:
    """直接获取数据库会话（用于非上下文管理器场景）"""
    return async_session()
