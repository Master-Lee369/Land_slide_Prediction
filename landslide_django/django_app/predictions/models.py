"""
django_app/predictions/models.py

Database models for the Landslide Detection Django app.
Stores uploaded images, prediction results, and analysis metadata.
"""

from django.db import models
import json


class LandslideAnalysis(models.Model):
    """Stores a single landslide analysis request and its results."""

    STATUS_CHOICES = [
        ('pending',    'Pending'),
        ('processing', 'Processing'),
        ('completed',  'Completed'),
        ('failed',     'Failed'),
    ]

    RISK_CHOICES = [
        ('low',      'Low Risk'),
        ('moderate', 'Moderate Risk'),
        ('high',     'High Risk'),
        ('critical', 'Critical Risk'),
    ]

    # ── Input metadata ──────────────────────────────────────────────────────
    location_name  = models.CharField(max_length=255, blank=True, default='')
    latitude       = models.FloatField(null=True, blank=True)
    longitude      = models.FloatField(null=True, blank=True)
    event_date     = models.DateField(null=True, blank=True)
    description    = models.TextField(blank=True, default='')

    # ── Uploaded image ───────────────────────────────────────────────────────
    image          = models.ImageField(upload_to='uploads/%Y/%m/')
    image_filename = models.CharField(max_length=255, blank=True)

    # ── ML feature inputs (user-provided or auto-extracted) ──────────────────
    ndvi           = models.FloatField(null=True, blank=True, help_text='Normalized Difference Vegetation Index (-1 to 1)')
    ndvi_change    = models.FloatField(null=True, blank=True, help_text='NDVI change pre/post event')
    slope_mean     = models.FloatField(null=True, blank=True, help_text='Mean slope in degrees')
    brightness     = models.FloatField(null=True, blank=True, help_text='Image brightness value')
    ratio_rg_change= models.FloatField(null=True, blank=True, help_text='Red/Green band ratio change')
    b3             = models.FloatField(null=True, blank=True, help_text='Sentinel-2 Band 3 (Green) mean value')

    # ── Status ───────────────────────────────────────────────────────────────
    status         = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at     = models.DateTimeField(auto_now_add=True)
    updated_at     = models.DateTimeField(auto_now=True)

    # ── Prediction results ───────────────────────────────────────────────────
    landslide_probability   = models.FloatField(null=True, blank=True, help_text='0.0–1.0 probability')
    risk_level              = models.CharField(max_length=20, choices=RISK_CHOICES, blank=True)
    predicted_class         = models.IntegerField(null=True, blank=True, help_text='1=landslide, 0=non-landslide')
    confidence_score        = models.FloatField(null=True, blank=True)

    # ── Visualization outputs ─────────────────────────────────────────────────
    segmentation_image      = models.ImageField(upload_to='results/segmentation/%Y/%m/', null=True, blank=True)
    heatmap_image           = models.ImageField(upload_to='results/heatmaps/%Y/%m/', null=True, blank=True)
    feature_chart_image     = models.ImageField(upload_to='results/charts/%Y/%m/', null=True, blank=True)

    # ── Feature importance stored as JSON ─────────────────────────────────────
    feature_importance_json = models.TextField(blank=True, default='{}')
    error_message           = models.TextField(blank=True, default='')

    class Meta:
        ordering = ['-created_at']
        verbose_name        = 'Landslide Analysis'
        verbose_name_plural = 'Landslide Analyses'

    def __str__(self):
        return f"Analysis #{self.pk} — {self.location_name or 'Unknown'} [{self.status}]"

    @property
    def feature_importance(self):
        try:
            return json.loads(self.feature_importance_json)
        except Exception:
            return {}

    @feature_importance.setter
    def feature_importance(self, value):
        self.feature_importance_json = json.dumps(value)

    def get_risk_color(self):
        return {
            'low':      '#22c55e',
            'moderate': '#f59e0b',
            'high':     '#ef4444',
            'critical': '#7c3aed',
        }.get(self.risk_level, '#6b7280')