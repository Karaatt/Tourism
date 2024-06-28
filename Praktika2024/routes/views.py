# routes/views.py

from django.shortcuts import render, get_object_or_404
from .models import Attraction
from .tourism_nn import predict_route

def home(request):
    attractions = Attraction.objects.all()

    if request.method == 'POST':
        start_point_id = request.POST.get('start_point')
        time_limit = request.POST.get('time_limit')

        if not start_point_id or not time_limit:
            return render(request, 'home.html', {
                'error': 'Please select a start point and enter a valid time limit.',
                'attractions': attractions
            })

        try:
            start_point = Attraction.objects.get(pk=start_point_id)

            predicted_route, total_time, total_distance = predict_route(start_point.name, int(time_limit))

            context = {
                'route': predicted_route,
                'total_time': total_time,
                'total_distance': total_distance,
                'attractions': attractions,  # Передача всех достопримечательностей для выпадающего списка
                'selected_start_point': start_point.name  # Передача выбранной локации для отображения
            }
            return render(request, 'home.html', context)
        except Attraction.DoesNotExist:
            return render(request, 'home.html', {
                'error': 'Selected start point does not exist.',
                'attractions': attractions
            })
        except IndexError as e:
            return render(request, 'home.html', {
                'error': str(e),
                'attractions': attractions
            })

    # Если запрос методом GET, просто отобразить форму с выпадающим списком
    return render(request, 'home.html', {'attractions': attractions})
