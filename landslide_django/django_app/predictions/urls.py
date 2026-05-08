"""
django_app/predictions/urls.py
"""
from django.urls import path
from . import views

urlpatterns = [
    path('analyze/',              views.AnalyzeView.as_view(),       name='analyze'),
    path('analyses/',             views.AnalysisListView.as_view(),   name='analysis-list'),
    path('analyses/<int:pk>/',    views.AnalysisDetailView.as_view(), name='analysis-detail'),
    path('stats/',                views.StatsView.as_view(),          name='stats'),
    path('fetch-satellite/',          views.FetchSatelliteView.as_view(),    name='fetch-satellite'),  
]