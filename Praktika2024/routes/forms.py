from django import forms
from .models import Attraction

class RouteForm(forms.Form):
    start_point = forms.ModelChoiceField(queryset=Attraction.objects.all(), label='Начальная точка')
    time_limit = forms.IntegerField(label='Количество времени (в часах)', min_value=1)
