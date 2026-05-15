import json
import csv

with open("../../data/crimescene/murdermysterylog.json", "r", encoding="utf-8") as f:
    games = json.load(f)

with open("../../data/crimescene/murdermysterylog.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    
    # 헤더
    writer.writerow(["url", "name", "rating", "players", "play_time", "description", "시리즈", "제작", "출판사", "국내 출판사", "reviews"])
    
    for g in games:
        tags = g.get("tags", {})
        writer.writerow([
            g.get("url", ""),
            g.get("name", ""),
            g.get("rating", ""),
            g.get("players", ""),
            g.get("play_time", ""),
            g.get("description", ""),
            tags.get("시리즈", ""),
            tags.get("제작", ""),
            # 출판사 복수면 | 로 구분
            " | ".join(tags["출판사"]) if isinstance(tags.get("출판사"), list) else tags.get("출판사", ""),
            tags.get("국내 출판사", ""),
            # 리뷰 여러 개면 || 로 구분
            " || ".join(g.get("reviews", [])),
        ])

print("완료! games.csv 저장됨")