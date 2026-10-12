from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parents[1]
PROJECT_ROOT = EVAL_ROOT.parent
BACKEND = PROJECT_ROOT / "backend"
RESULT_DIR = EVAL_ROOT / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_SUFFIXES = {".py", ".js", ".ts", ".vue", ".json", ".md", ".txt", ".yml", ".yaml"}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def find_files(pattern: str):
    return list(PROJECT_ROOT.rglob(pattern))


def any_regex(files, pattern: str, flags=re.I | re.S):
    rx = re.compile(pattern, flags)
    hits = []
    for path in files:
        text = read_text(path)
        if rx.search(text):
            hits.append(str(path.relative_to(PROJECT_ROOT)))
    return hits


def all_source_files():
    out = []
    for path in PROJECT_ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            parts = set(path.parts)
            excluded_dirs = {
                ".venv", "node_modules", "__pycache__", ".git",
                "MoonCAREpack", "dist", "build",
            }
            if parts & excluded_dirs:
                continue
            out.append(path)
    return out


def git_tracked_env():
    try:
        p = subprocess.run(
            ["git", "ls-files"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if p.returncode != 0:
            return []
        tracked = []
        for line in p.stdout.splitlines():
            name = Path(line).name.lower()
            if name == ".env":
                tracked.append(line)
            elif name.startswith(".env.") and name not in {
                ".env.example", ".env.sample", ".env.template",
            }:
                tracked.append(line)
        return tracked
    except Exception:
        return []


def main():
    source_files = all_source_files()
    agent_service = BACKEND / "app" / "services" / "agent_service.py"
    diary_api = BACKEND / "app" / "api" / "v1" / "diary.py"
    user_model = BACKEND / "app" / "models" / "user.py"

    agent_text = read_text(agent_service)
    diary_text = read_text(diary_api)
    user_text = read_text(user_model)

    checks = []

    def add(cid, category, title, status, evidence, severity="info"):
        checks.append({
            "id": cid,
            "category": category,
            "title": title,
            "status": status,
            "severity": severity,
            "evidence": evidence,
        })

    # 1) Structured AgentTrace uses length rather than raw message in request_start.
    if "message_length=len(user_message or \"\")" in agent_text or "message_length=len(user_message or '')" in agent_text:
        add("S14-001", "logging", "AgentTrace 请求日志避免直接记录原始消息", "pass",
            "agent_service.py request_start uses message_length instead of raw user_message")
    else:
        add("S14-001", "logging", "AgentTrace 请求日志避免直接记录原始消息", "review",
            "未找到 message_length 形式，请人工复核 AgentTrace", "medium")

    # 2) Scan logger/print calls that may include raw user_message/message content.
    suspicious = []
    raw_patterns = [
        r"(logger\.(?:info|warning|error|debug|exception)|print)\([^\n]{0,220}user_message",
        r"(logger\.(?:info|warning|error|debug|exception)|print)\([^\n]{0,220}original_text",
    ]
    for p in raw_patterns:
        suspicious.extend(any_regex(source_files, p, re.I))
    suspicious = sorted(set(suspicious))
    if not suspicious:
        add("S14-002", "logging", "源码扫描未发现明显原始对话/日记正文日志", "pass",
            "No obvious logger/print call with user_message or original_text found")
    else:
        add("S14-002", "logging", "原始敏感文本日志扫描", "review",
            "需人工复核: " + ", ".join(suspicious[:8]), "medium")

    # 3) Diary read ownership check.
    has_read_owner = (
        "MoodDiary.id == diary_id" in diary_text
        and "MoodDiary.user_id == user_id" in diary_text
    )
    add("S14-003", "authorization", "日记读取按 diary_id + user_id 做所有权过滤",
        "pass" if has_read_owner else "review",
        "diary.py contains both MoodDiary.id == diary_id and MoodDiary.user_id == user_id"
        if has_read_owner else "未能确认日记读取所有权过滤", "high" if not has_read_owner else "info")

    # 4) Diary delete ownership check.
    delete_block = re.search(
        r"@router\.delete\([^\n]*\).*?def\s+delete_mood_diary.*?(?=@router\.|\Z)",
        diary_text,
        re.S,
    )
    delete_text = delete_block.group(0) if delete_block else ""
    has_delete_owner = (
        "MoodDiary.id == diary_id" in delete_text
        and "MoodDiary.user_id == user_id" in delete_text
        and "db.delete(diary)" in delete_text
    )
    add("S14-004", "deletion", "单条情绪日记删除包含用户所有权校验",
        "pass" if has_delete_owner else "review",
        "delete_mood_diary filters by diary_id + user_id before db.delete"
        if has_delete_owner else "未能确认完整删除所有权校验", "high" if not has_delete_owner else "info")

    # 5) User-owned relationship cascade evidence.
    cascade_hits = []
    for line in user_text.splitlines():
        if "relationship(" in line and "cascade" in line:
            cascade_hits.append(line.strip())
    add("S14-005", "deletion", "User 模型存在级联关系配置",
        "pass" if cascade_hits else "review",
        " | ".join(cascade_hits[:6]) if cascade_hits else "未找到 cascade relationship；需人工核对",
        "medium" if not cascade_hits else "info")

    # 6) Search whole-account deletion API.
    account_delete_patterns = [
        r"@router\.delete\([^\n]*(?:account|user|me|profile)",
        r"def\s+(?:delete_account|delete_user|remove_account)",
    ]
    account_hits = []
    for p in account_delete_patterns:
        account_hits.extend(any_regex(source_files, p, re.I))
    account_hits = sorted(set(account_hits))
    add("S14-006", "deletion", "整账号数据删除入口",
        "pass" if account_hits else "gap",
        ", ".join(account_hits[:8]) if account_hits else "未发现明确的整账号删除 API；当前只能确认局部数据删除能力",
        "high" if not account_hits else "info")

    # 7) Search conversation deletion.
    conv_delete_hits = any_regex(
        source_files,
        r"@router\.delete\([^\n]*conversation|def\s+delete_[a-z_]*conversation",
        re.I,
    )
    add("S14-007", "deletion", "会话记录删除入口",
        "pass" if conv_delete_hits else "gap",
        ", ".join(conv_delete_hits[:8]) if conv_delete_hits else "未发现明确会话删除 API",
        "medium" if not conv_delete_hits else "info")

    # 8) .env tracking.
    tracked_env = git_tracked_env()
    add("S14-008", "secrets", ".env 未被 Git 跟踪",
        "pass" if not tracked_env else "fail",
        "No tracked .env file" if not tracked_env else "Tracked secret-like files: " + ", ".join(tracked_env),
        "critical" if tracked_env else "info")

    # 9) Search ownership filters across diary/draft APIs.
    ownership_count = diary_text.count("user_id == user_id")
    add("S14-009", "authorization", "Diary/Draft API 广泛使用 user_id 所有权约束",
        "pass" if ownership_count >= 3 else "review",
        f"diary.py occurrences of 'user_id == user_id': {ownership_count}",
        "medium" if ownership_count < 3 else "info")

    # 10) Sensitive-domain data categories observable in models.
    model_dir = BACKEND / "app" / "models"
    model_text = "\n".join(read_text(p) for p in model_dir.glob("*.py"))
    categories = {
        "biometric": "Biometric" in model_text or "biometric" in model_text,
        "menstrual": "Menstrual" in model_text or "menstrual" in model_text,
        "mood_diary": "MoodDiary" in model_text or "mood_diary" in model_text,
        "conversation": "Conversation" in model_text or "conversation" in model_text,
        "music_feedback": "MusicFeedback" in model_text or "music_feedback" in model_text,
    }
    add("S14-010", "data_inventory", "敏感数据类别可从模型层建立数据字典",
        "pass" if sum(categories.values()) >= 3 else "review",
        json.dumps(categories, ensure_ascii=False),
        "info")

    pass_n = sum(c["status"] == "pass" for c in checks)
    gap_n = sum(c["status"] == "gap" for c in checks)
    fail_n = sum(c["status"] == "fail" for c in checks)
    review_n = sum(c["status"] == "review" for c in checks)

    summary = {
        "checks": len(checks),
        "passed": pass_n,
        "confirmed_gaps": gap_n,
        "failed": fail_n,
        "manual_review": review_n,
        "note": (
            "这是源码级隐私/删除能力审计，不等同于完整隐私合规或真实采集-存储-读取-删除闭环验证。"
        ),
    }

    with (RESULT_DIR / "privacy_audit_raw_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(checks[0].keys()))
        writer.writeheader()
        writer.writerows(checks)

    (RESULT_DIR / "privacy_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RESULT_DIR / "privacy_audit_gaps.json").write_text(
        json.dumps([c for c in checks if c["status"] in {"gap", "fail", "review"}],
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# MoonCARE S14 Privacy / Deletion Source Audit",
        "",
        f"- Checks: {len(checks)}",
        f"- Passed: {pass_n}",
        f"- Confirmed gaps: {gap_n}",
        f"- Failed: {fail_n}",
        f"- Manual review: {review_n}",
        "",
        "## Boundary",
        "",
        summary["note"],
        "",
        "不要把本结果写成“隐私合规已完成”。它只能证明当前源码中可观察到的",
        "日志最小化、局部所有权校验、局部删除能力与已发现的删除链缺口。",
    ]
    (RESULT_DIR / "privacy_audit_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("=" * 64)
    print("MoonCARE S14 Privacy / Deletion Source Audit")
    print("=" * 64)
    print(f"Checks: {len(checks)}")
    print(f"Passed: {pass_n}")
    print(f"Confirmed Gaps: {gap_n}")
    print(f"Failed: {fail_n}")
    print(f"Manual Review: {review_n}")
    print(f"Results written to: {RESULT_DIR}")


if __name__ == "__main__":
    main()
