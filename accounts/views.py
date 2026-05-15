from django.shortcuts import render, redirect
<<<<<<< HEAD
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import SignupForm, PersonaForm
from .models import Persona


def signup(request):
    if request.method == "POST":
        form = SignupForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "회원가입이 완료되었습니다!")
            return redirect("accounts:persona")
    else:
        form = SignupForm()
    return render(request, "accounts/signup.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect(request.GET.get("next", "/"))
        messages.error(request, "아이디 또는 비밀번호가 올바르지 않습니다.")
    return render(request, "accounts/login.html")


def logout_view(request):
    logout(request)
    return redirect("/")


@login_required
def persona(request):
    """페르소나 설정 / 수정"""
    instance, _ = Persona.objects.get_or_create(user=request.user)
    if request.method == "POST":
        form = PersonaForm(request.POST, instance=instance)
        if form.is_valid():
            form.save()
            messages.success(request, "성향이 저장되었습니다!")
            return redirect("/")
    else:
        form = PersonaForm(instance=instance)
    return render(request, "accounts/persona.html", {"form": form})