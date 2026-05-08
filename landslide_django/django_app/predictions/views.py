"""
django_app/predictions/views.py

API views:
  POST /api/analyze/          — upload image + metadata, run ML pipeline
  GET  /api/analyses/         — list all analyses
  GET  /api/analyses/<id>/    — detail of one analysis
  DELETE /api/analyses/<id>/  — delete an analysis
  GET  /api/stats/            — summary statistics
"""
from django.conf import settings
from rest_framework.views       import APIView
from rest_framework.response    import Response
from rest_framework             import status
from django.shortcuts           import get_object_or_404
from django.db.models           import Avg, Count

from .models      import LandslideAnalysis
from .serializers import LandslideAnalysisSerializer, LandslideAnalysisListSerializer
from .ml_bridge   import run_full_pipeline
from .fetch_satellite import fetch_satellite_image
import os

class AnalyzeView(APIView):
    """POST: Upload image + form data → run ML pipeline → return results."""

    def post(self, request):
        image = request.FILES.get('image')
        if not image:
            return Response({'error': 'No image file provided.'}, status=status.HTTP_400_BAD_REQUEST)

        # Build analysis instance from form data
        def _float(key):
            val = request.data.get(key)
            try:
                return float(val) if val not in (None, '') else None
            except (ValueError, TypeError):
                return None

        analysis = LandslideAnalysis(
            image          = image,
            image_filename = image.name,
            location_name  = request.data.get('location_name', ''),
            description    = request.data.get('description', ''),
            event_date     = request.data.get('event_date') or None,
            ndvi           = _float('ndvi'),
            b3             = _float('b3'),
            slope_mean     = _float('slope_mean'),
            brightness     = _float('brightness'),
            ndvi_change    = _float('ndvi_change'),
            ratio_rg_change= _float('ratio_rg_change'),
            status         = 'processing',
        )

        lat = _float('latitude')
        lon = _float('longitude')
        if lat is not None: analysis.latitude  = lat
        if lon is not None: analysis.longitude = lon

        analysis.save()

        # Run the ML pipeline synchronously
        analysis = run_full_pipeline(analysis)

        serializer = LandslideAnalysisSerializer(analysis, context={'request': request})
        http_status = status.HTTP_201_CREATED if analysis.status == 'completed' else status.HTTP_200_OK
        return Response(serializer.data, status=http_status)


class AnalysisListView(APIView):
    """GET: List all analyses (newest first)."""

    def get(self, request):
        analyses   = LandslideAnalysis.objects.all()
        serializer = LandslideAnalysisListSerializer(analyses, many=True, context={'request': request})
        return Response(serializer.data)


class AnalysisDetailView(APIView):
    """GET / DELETE: Single analysis detail."""

    def get(self, request, pk):
        analysis   = get_object_or_404(LandslideAnalysis, pk=pk)
        serializer = LandslideAnalysisSerializer(analysis, context={'request': request})
        return Response(serializer.data)

    def delete(self, request, pk):
        analysis = get_object_or_404(LandslideAnalysis, pk=pk)
        analysis.delete()
        return Response({'message': 'Analysis deleted.'}, status=status.HTTP_204_NO_CONTENT)


class StatsView(APIView):
    """GET: Summary statistics for the dashboard."""

    def get(self, request):
        qs    = LandslideAnalysis.objects.filter(status='completed')
        total = LandslideAnalysis.objects.count()

        risk_counts = dict(
            qs.values_list('risk_level').annotate(c=Count('id')).values_list('risk_level', 'c')
        )

        avg_prob = qs.aggregate(avg=Avg('landslide_probability'))['avg']

        recent = LandslideAnalysis.objects.order_by('-created_at')[:5]
        recent_data = LandslideAnalysisListSerializer(recent, many=True, context={'request': request}).data

        return Response({
            'total_analyses':       total,
            'completed':            qs.count(),
            'avg_probability':      round(avg_prob, 3) if avg_prob else 0,
            'risk_distribution':    risk_counts,
            'landslide_detected':   qs.filter(predicted_class=1).count(),
            'recent_analyses':      recent_data,
        })


class FetchSatelliteView(APIView):

    def get(self, request):
        lat = request.query_params.get('lat')
        lon = request.query_params.get('lon')

        if not lat or not lon:
            return Response(
                {'error': 'lat and lon query parameters are required.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except ValueError:
            return Response(
                {'error': 'lat and lon must be valid numbers.'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            import traceback
            from .fetch_satellite import fetch_satellite_image
            from .ml_bridge import extract_features_from_image   # ← extract features

            # 1. Fetch satellite image
            img_path, scene_name = fetch_satellite_image(lat_f, lon_f)

            # 2. Auto-extract ML features from the fetched image
            features = extract_features_from_image(img_path)

            # 3. Build media URL
            media_root = str(settings.MEDIA_ROOT)
            relative   = img_path.replace(media_root, '').replace('\\', '/').lstrip('/')
            img_url    = request.build_absolute_uri(settings.MEDIA_URL + relative)

            return Response({
                'image_path':  img_path,
                'image_url':   img_url,
                'scene_name':  scene_name,
                'latitude':    lat_f,
                'longitude':   lon_f,
                # ← these are the auto-extracted feature values
                'ndvi':             features['ndvi'],
                'ndvi_change':      features['ndvi_change'],
                'b3':               features['b3'],
                'slope_mean':       features['slope_mean'],
                'brightness':       features['brightness'],
                'ratio_rg_change':  features['ratio_rg_change'],
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)