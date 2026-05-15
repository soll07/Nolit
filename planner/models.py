from django.db import models
from django.conf import settings


class PlannerSession(models.Model):
    """
    플래너 세션 - 유저가 구성한 게임 조합 저장
    """

    user       = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        null=True, blank=True,          # 비로그인도 사용 가능
        related_name="planner_sessions",
        verbose_name="유저",
    )
    title      = models.CharField(max_length=100, blank=True, verbose_name="세션 이름")
    players    = models.PositiveIntegerField(null=True, blank=True, verbose_name="인원 수")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table     = "planner_session"
        verbose_name = "플래너 세션"
        ordering     = ["-created_at"]

    def __str__(self):
        return f"{self.title or '제목 없음'} ({self.players}명)"

    @property
    def total_time(self):
        """세션에 담긴 게임들의 총 플레이 시간(분) 합산"""
        return sum(item.play_time for item in self.items.all() if item.play_time)


class PlannerItem(models.Model):
    """
    플래너 세션에 담긴 개별 게임 항목
    """

    CATEGORY_CHOICES = [
        ("boardgame",    "보드게임"),
        ("escape",       "방탈출"),
        ("murdermystery","머더미스터리"),
    ]

    session    = models.ForeignKey(
        PlannerSession,
        on_delete=models.CASCADE,
        related_name="items",
        verbose_name="플래너 세션",
    )
    category   = models.CharField(max_length=20, choices=CATEGORY_CHOICES, verbose_name="카테고리")
    game_id    = models.PositiveIntegerField(verbose_name="게임 ID")
    game_name  = models.CharField(max_length=200, verbose_name="게임명")
    play_time  = models.PositiveIntegerField(null=True, blank=True, verbose_name="플레이 시간(분)")
    players    = models.CharField(max_length=20, blank=True, verbose_name="인원")
    order      = models.PositiveIntegerField(default=0, verbose_name="순서")

    class Meta:
        db_table     = "planner_item"
        verbose_name = "플래너 아이템"
        ordering     = ["order"]

    def __str__(self):
        return f"{self.game_name} ({self.play_time}분)"