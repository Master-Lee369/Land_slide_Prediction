"""
django_app/predictions/admin.py
"""
from django.contrib import admin
from .models import LandslideAnalysis


@admin.register(LandslideAnalysis)
class LandslideAnalysisAdmin(admin.ModelAdmin):
    list_display  = ['id', 'location_name', 'status', 'risk_level',
                     'landslide_probability', 'confidence_score', 'created_at']
    list_filter   = ['status', 'risk_level']
    search_fields = ['location_name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'feature_importance_json',
                       'error_message', 'segmentation_image', 'heatmap_image',
                       'feature_chart_image']
    fieldsets = (
        ('Location & Event',    {'fields': ('location_name', 'latitude', 'longitude', 'event_date', 'description')}),
        ('Image',               {'fields': ('image', 'image_filename')}),
        ('ML Features',         {'fields': ('ndvi', 'b3', 'slope_mean', 'brightness', 'ndvi_change', 'ratio_rg_change')}),
        ('Results',             {'fields': ('status', 'predicted_class', 'landslide_probability', 'risk_level', 'confidence_score')}),
        ('Visualisations',      {'fields': ('heatmap_image', 'segmentation_image', 'feature_chart_image')}),
        ('Debug',               {'fields': ('feature_importance_json', 'error_message', 'created_at', 'updated_at'),
                                 'classes': ('collapse',)}),
    )