<<<<<<< HEAD
from .yoonha_graph import graph

import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator


def home(request):
    return render(request, "index.html")


@method_decorator(csrf_exempt, name="dispatch")
class ChatView(View):

    def get(self, request):
        return render(request, "recommender/chat.html")

    def post(self, request):
        try:
            body    = json.loads(request.body)
            message = body.get("message", "").strip()
            if not message:
                return JsonResponse({"error": "메시지를 입력해주세요."}, status=400)
        except json.JSONDecodeError:
            return JsonResponse({"error": "잘못된 요청 형식입니다."}, status=400)

        history = request.session.get("chat_history", [])
        group   = request.session.get("group", {})

        try:
            from .yoonha_graph import graph
            result = graph.invoke({
                "query":    message,
                "category": body.get("category", ""),
                "group":    group,
                "use_api":  True,
            })
            answer        = result.get("answer", "")
            games         = result.get("games", [])
            next_question = result.get("next_question", "")
        except Exception as e:
            answer        = f"오류가 발생했습니다: {str(e)}"
            games         = []
            next_question = ""

        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant",  "content": answer})
        request.session["chat_history"] = history[-20:]

        return JsonResponse({
            "answer":        answer,
            "games":         games,
            "next_question": next_question,
        })

    def delete(self, request):
        request.session.pop("chat_history", None)
        request.session.pop("group", None)
        return JsonResponse({"message": "대화가 초기화되었습니다."})


@method_decorator(csrf_exempt, name="dispatch")
class RecommendView(View):

    def post(self, request):
        try:
            body     = json.loads(request.body)
            query    = body.get("query", "")
            category = body.get("category", "")
            players  = body.get("players", "")
            time     = body.get("time", "")
        except json.JSONDecodeError:
            return JsonResponse({"error": "잘못된 요청 형식입니다."}, status=400)

        try:
            from .yoonha_graph import graph
            result = graph.invoke({
                "query":    f"{query} 인원:{players} 시간:{time}",
                "category": category,
                "group":    {"headcount": players, "max_time": time},
                "use_api":  True,
            })
            answer = result.get("answer", "")
            games  = result.get("games", [])
        except Exception as e:
            answer = f"오류가 발생했습니다: {str(e)}"
            games  = []

        return JsonResponse({"answer": answer, "games": games})