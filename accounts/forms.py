from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User, Persona


class SignupForm(UserCreationForm):
    email    = forms.EmailField(required=True)
    nickname = forms.CharField(max_length=50, required=False, label="닉네임")

    class Meta:
        model  = User
        fields = ("username", "email", "nickname", "password1", "password2")


class PersonaForm(forms.ModelForm):
    class Meta:
        model  = Persona
        fields = ("preferred_category", "group_size", "max_playtime", "experience", "notes")
        labels = {
            "preferred_category": "주로 즐기는 장르",
            "group_size":         "평소 인원 수",
            "max_playtime":       "선호 최대 시간(분)",
            "experience":         "경험 수준",
            "notes":              "추가 성향",
        }
        widgets = {
            "notes": forms.Textarea(attrs={"rows": 3}),
        }