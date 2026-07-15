"""
一次性清理脚本：删除“主题/科目被错误按字符拆分”产生的旧资源。

背景：extraction 修复之前，像“数字逻辑”这样的复合词会被拆成
subject="数"、topic="字逻辑"，导致标题变成“数·字逻辑 XXX”。
这些脏数据已经写入数据库，代码层面的修复无法追溯改正已存在的记录，
需要跑一次本脚本清理。

用法（在项目根目录，和 main.py / models.py 放在一起）：
    python cleanup_bad_resources.py            # 先预览会删除哪些记录
    python cleanup_bad_resources.py --apply     # 确认后真正执行删除

判定规则（保守）：subject 或 topic 的长度为 1 个字符，
这是正常学科/主题名几乎不可能出现的情况，大概率是错误切分的产物。
如果你想手动指定要删除的具体 subject/topic，也可以用 --subject/--topic。
"""
import asyncio
import argparse

from sqlalchemy import select, delete
from models import async_session, LearningResource, LearningPath


async def main(apply: bool, subject: str = None, topic: str = None):
    async with async_session() as db:
        result = await db.execute(select(LearningResource))
        all_resources = result.scalars().all()

        if subject or topic:
            targets = [
                r for r in all_resources
                if (subject is None or r.subject == subject)
                and (topic is None or r.topic == topic)
            ]
        else:
            targets = [
                r for r in all_resources
                if len((r.subject or "").strip()) <= 1 or len((r.topic or "").strip()) <= 1
            ]

        if not targets:
            print("没有找到符合条件的可疑记录。")
            return

        print(f"匹配到 {len(targets)} 条待清理记录：")
        seen = set()
        for r in targets:
            key = (r.subject, r.topic)
            if key not in seen:
                seen.add(key)
                print(f"  - subject={r.subject!r} topic={r.topic!r}")

        if not apply:
            print("\n[预览模式] 未执行删除。确认无误后请加 --apply 参数重新运行。")
            return

        ids = [r.id for r in targets]
        await db.execute(delete(LearningResource).where(LearningResource.id.in_(ids)))

        # 同步清理对应的学习路径
        for s, t in seen:
            await db.execute(
                delete(LearningPath).where(
                    LearningPath.subject == s,
                    LearningPath.title.like(f"%{t}%"),
                )
            )

        await db.commit()
        print(f"\n已删除 {len(ids)} 条资源记录及相关学习路径。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="真正执行删除（默认只预览）")
    parser.add_argument("--subject", default=None, help="只清理指定 subject 的记录")
    parser.add_argument("--topic", default=None, help="只清理指定 topic 的记录")
    args = parser.parse_args()
    asyncio.run(main(args.apply, args.subject, args.topic))
