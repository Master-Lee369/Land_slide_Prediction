"""
django_app/predictions/serializers.py
"""

from rest_framework import serializers
from .models import LandslideAnalysis


class LandslideAnalysisSerializer(serializers.ModelSerializer):
    """Full detail serializer — used for create and retrieve."""

    image_url              = serializers.SerializerMethodField()
    heatmap_url            = serializers.SerializerMethodField()
    segmentation_url       = serializers.SerializerMethodField()
    feature_chart_url      = serializers.SerializerMethodField()
    feature_importance     = serializers.SerializerMethodField()
    risk_color             = serializers.SerializerMethodField()

    class Meta:
        model  = LandslideAnalysis
        fields = [
            'id', 'location_name', 'latitude', 'longitude',
            'event_date', 'description', 'image_filename',
            'ndvi', 'b3', 'slope_mean', 'brightness', 'ndvi_change', 'ratio_rg_change',
            'status', 'created_at', 'updated_at',
            'landslide_probability', 'risk_level', 'predicted_class', 'confidence_score',
            'feature_importance', 'error_message',
            'image_url', 'heatmap_url', 'segmentation_url', 'feature_chart_url',
            'risk_color',
        ]

    def _build_url(self, request, file_field):
        if file_field and hasattr(file_field, 'url'):
            try:
                return request.build_absolute_uri(file_field.url)
            except Exception:
                return None
        return None

    def get_image_url(self, obj):
        return self._build_url(self.context.get('request'), obj.image)

    def get_heatmap_url(self, obj):
        return self._build_url(self.context.get('request'), obj.heatmap_image)

    def get_segmentation_url(self, obj):
        return self._build_url(self.context.get('request'), obj.segmentation_image)

    def get_feature_chart_url(self, obj):
        return self._build_url(self.context.get('request'), obj.feature_chart_image)

    def get_feature_importance(self, obj):
        return obj.feature_importance

    def get_risk_color(self, obj):
        return obj.get_risk_color()


class LandslideAnalysisListSerializer(serializers.ModelSerializer):
    """Lightweight list serializer — used for history table."""

    image_url  = serializers.SerializerMethodField()
    risk_color = serializers.SerializerMethodField()

    class Meta:
        model  = LandslideAnalysis
        fields = [
            'id', 'location_name', 'event_date', 'status',
            'landslide_probability', 'risk_level', 'risk_color',
            'predicted_class', 'confidence_score', 'created_at', 'image_url',
        ]

    def get_image_url(self, obj):
        req = self.context.get('request')
        if obj.image and req:
            try:
                return req.build_absolute_uri(obj.image.url)
            except Exception:
                return None
        return None

    def get_risk_color(self, obj):
        return obj.get_risk_color()