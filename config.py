"""
高等教育个性化学习资源智能体系统 - 全局配置
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent

# 加载.env文件
env_file = BASE_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)


class Settings:
    """系统配置"""
    # 项目信息
    PROJECT_NAME: str = "高等教育个性化学习资源智能体系统"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = "基于多智能体架构的个性化学习资源生成与管理系统"

    # LLM配置
    LLM_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 4096

    # 嵌入模型配置
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # 数据库配置
    DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "mysql+aiomysql://admin:admin123@10.78.202.91:3306/edu_agent"
    )

    # 文件存储
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    RESOURCE_DIR: Path = BASE_DIR / "resources"
    TEMPLATE_DIR: Path = BASE_DIR / "templates"
    STATIC_DIR: Path = BASE_DIR / "static"

    # 服务器配置
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # 画像维度配置（不少于6个维度）
    PROFILE_DIMENSIONS: list = [
        "知识基础",
        "认知风格",
        "学习偏好",
        "易错点偏好",
        "学习节奏",
        "目标导向",
        "学科兴趣",
        "学习习惯"
    ]

    # 智能体配置
    AGENT_TYPES: list = [
        "profile_agent",        # 画像构建智能体
        "content_agent",        # 内容生成智能体
        "exercise_agent",       # 习题生成智能体（含自动批阅）
        "mindmap_agent",        # 思维导图智能体
        "ppt_slides_agent",     # 课件讲义智能体
        "code_agent",           # 代码实操智能体
        "reading_agent",        # 拓展阅读智能体
        "path_agent",           # 路径规划智能体
        "tutor_agent",          # 辅导答疑智能体
        "evaluation_agent",     # 评估智能体
        "orchestrator_agent",   # 协调智能体
    ]

    # 资源类型
    RESOURCE_TYPES: list = [
        "lecture_doc",          # 课程讲解文档
        "mind_map",             # 思维导图
        "exercise",             # 练习题
        "reading_material",     # 拓展阅读
        "ppt_slides",           # 课件讲义
        "code_practice",        # 代码实操案例
        "project_case",         # 实践项目
    ]

    def __init__(self):
        # 确保目录存在
        for dir_path in [self.UPLOAD_DIR, self.RESOURCE_DIR,
                          self.TEMPLATE_DIR, self.STATIC_DIR,
                          BASE_DIR / "data"]:
            dir_path.mkdir(parents=True, exist_ok=True)


settings = Settings()
