from django.db import models

from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    nickname = models.CharField(max_length=50, blank=True)


class Persona(models.Model):
    CATEGORY_CHOICES = [
        ("boardgame",    "보드게임"),
        ("escape",       "방탈출"),
        ("murdermystery","머더미스터리"),
    ]

    user               = models.OneToOneField(User, on_delete=models.CASCADE, related_name="persona")
    preferred_category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, blank=True)
    group_size         = models.PositiveIntegerField(null=True, blank=True)
    max_playtime       = models.PositiveIntegerField(null=True, blank=True)
    experience         = models.CharField(
        max_length=20,
        choices=[("beginner","입문"), ("intermediate","중급"), ("expert","고수")],
        default="beginner",
    )
    notes      = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)