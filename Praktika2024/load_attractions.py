import pandas as pd
import os
import django

# Настройка Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Praktika2024.settings")
django.setup()

from routes.models import Attraction

# Загрузка данных из Excel файлов
df_attractions = pd.read_excel('stavropol_attractions.xlsx')
df_coords = pd.read_excel('stavropol_attractions_coords__kopia_1.xlsx')

# Заполнение базы данных из Excel файла
for index, row in df_attractions.iterrows():
    matching_coords = df_coords[df_coords['Название'] == row['Название']]
    working_hours = matching_coords['Время работы'].values[0] if not matching_coords.empty else None

    Attraction.objects.create(
        name=row['Название'],
        address=row['Адрес'],
        rating=row.get('Рейтинг', None),
        working_hours=working_hours,
        latitude=None,  # Заполните, если есть соответствующие данные
        longitude=None  # Заполните, если есть соответствующие данные
    )
