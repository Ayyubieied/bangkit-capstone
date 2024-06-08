import datetime
import json
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.http import HttpResponseRedirect, JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django import forms

class UserForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(max_length=150)

@csrf_exempt
def login_user(request):
    if request.method == "POST":
        data = json.loads(request.body)
        username = data.get("username")
        password = data.get("password")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            response = JsonResponse({
                "username": username,
                "status": True,
                "message": "Successfully Logged In!"
            }, status=200)
            response.set_cookie("last_login", str(datetime.datetime.now()))
            return response
        else:
            return JsonResponse({
                "status": False,
                "message": "Failed to Login, Account Disabled."
            }, status=401)

@csrf_exempt
def register(request):
    if request.method == "POST":
        data = json.loads(request.body)
        form = UserForm(data)
        if form.is_valid():
            username = form.cleaned_data['username']
            if User.objects.filter(username=username).exists():
                return JsonResponse({
                    "status": False,
                    'message': 'Username is already taken',
                }, status=400)
            else:
                User.objects.create_user(
                    username=username, 
                    password=form.cleaned_data['password']
                )
                return JsonResponse({
                    "status": True,
                    'message': 'User successfully registered',
                }, status=200)
        else:
            return JsonResponse({
                "status": False,
                'message': 'Something went wrong',
                'errors': form.errors,
            }, status=400)
    
    return JsonResponse({
        "status": False,
        'message': 'Invalid request method',
    }, status=405)
    
@csrf_exempt
def logout_user(request):
    if request.user.is_authenticated:
        logout(request)
        return JsonResponse({
            "status": True,
            "message": "Successfully logged out"
        }, status=200)
    else:
        return JsonResponse({
            "status": False,
            "message": "User is not logged in"
        }, status=400)