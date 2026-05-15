# 게임 조합 플레이타임 계산 로직

def calculate_total_time(games: list[dict]) -> dict:
    """
    선택된 게임들의 총 플레이 시간 계산

    Args:
        games: [{"name": "...", "play_time": 60, "players": "2~4"}, ...]

    Returns:
        {
            "total_min": 180,        # 최소 예상 시간(분)
            "total_max": 240,        # 최대 예상 시간(분)
            "setup_time": 15,        # 셋업 시간(분) - 게임당 5분 기준
            "breakdown": [...]       # 게임별 상세
        }
    """
    breakdown  = []
    total_min  = 0
    total_max  = 0
    setup_time = len(games) * 5  # 게임당 셋업 5분 가정

    for game in games:
        play_time = game.get("play_time", 0)

        # play_time이 "60분" 문자열이면 숫자 추출
        if isinstance(play_time, str):
            import re
            m = re.search(r"\d+", play_time)
            play_time = int(m.group()) if m else 0

        # 범위 계산 (±20% 편차)
        time_min = int(play_time * 0.9)
        time_max = int(play_time * 1.2)

        total_min += time_min
        total_max += time_max

        breakdown.append({
            "name":     game.get("name", ""),
            "time_min": time_min,
            "time_max": time_max,
        })

    return {
        "total_min":  total_min + setup_time,
        "total_max":  total_max + setup_time,
        "setup_time": setup_time,
        "breakdown":  breakdown,
    }


def check_player_compatibility(games: list[dict], target_players: int) -> list[dict]:
    """
    선택된 게임이 인원 수와 호환되는지 확인

    Returns:
        [{"name": "...", "compatible": True/False, "reason": "..."}, ...]
    """
    import re
    results = []

    for game in games:
        players_str = game.get("players", "")
        compatible  = True
        reason      = ""

        m = re.search(r"(\d+)\s*[~～]\s*(\d+)", players_str)
        if m:
            min_p, max_p = int(m.group(1)), int(m.group(2))
            if target_players < min_p:
                compatible = False
                reason = f"최소 {min_p}명 필요"
            elif target_players > max_p:
                compatible = False
                reason = f"최대 {max_p}명까지 가능"
        else:
            m2 = re.search(r"(\d+)", players_str)
            if m2 and int(m2.group(1)) != target_players:
                compatible = False
                reason = f"정확히 {m2.group(1)}명 필요"

        results.append({
            "name":       game.get("name", ""),
            "compatible": compatible,
            "reason":     reason,
        })

    return results