import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Загрузка датасета
attractions_df = pd.read_excel('stavropol_attractions_coords — копия.xlsx')

# Ограничение количества мест до 50
attractions_df = attractions_df.sort_values(by='Рейтинг').head(50).reset_index()

# Загрузка матриц расстояний и времени
distance_matrix = np.load('distance_matrix.npy')
time_matrix = np.load('time_matrix.npy')

# Параметры
time_per_location = 20 / 60  # 20 минут в часах
max_time = 24  # 24 часа

# Parse working hours
def parse_working_hours(time_str):
    try:
        start_time, end_time = time_str.split('–')
        start_hour, start_minute = map(int, start_time.split(':'))
        end_hour, end_minute = map(int, end_time.split(':'))
        start_time_in_hours = start_hour + start_minute / 60
        end_time_in_hours = end_hour + end_minute / 60
        return start_time_in_hours, end_time_in_hours
    except ValueError:
        return 0, 24  # Default to open all day if parsing fails

# Add working hours to attractions dataframe
attractions_df['Working Hours'] = attractions_df['Время работы'].apply(parse_working_hours)

# Определение модели
class RouteOptimizerNN(nn.Module):
    def __init__(self, n_locations):
        super(RouteOptimizerNN, self).__init__()
        self.fc1 = nn.Linear(2 * n_locations, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Создание модели
model = RouteOptimizerNN(n_locations=len(attractions_df))

# Загрузка сохраненной модели
model.load_state_dict(torch.load('route_optimizer_model.pth'))
model.eval()

# Предсказание маршрута
def predict_route(model, attractions_df, time_matrix, max_time, time_per_location):
    n_locations = len(attractions_df)
    visited = [False] * n_locations
    route = []
    total_time = 0
    total_distance = 0
    current_time = 0  # start at 0:00

    current_location = None

    # Find the first location to visit that is open at the start
    for i in range(n_locations):
        start_working_time, end_working_time = attractions_df.loc[i, 'Working Hours']
        if start_working_time <= current_time <= end_working_time:
            current_location = i
            break

    if current_location is None:
        earliest_open_time = min(attractions_df['Working Hours'].apply(lambda x: x[0]))
        current_location = attractions_df['Working Hours'].apply(lambda x: x[0]).idxmin()
        current_time = earliest_open_time

    while total_time + time_per_location <= max_time:
        start_working_time, end_working_time = attractions_df.loc[current_location, 'Working Hours']

        route.append(current_location)
        visited[current_location] = True
        total_time += time_per_location
        current_time += time_per_location

        next_location = None
        min_time = float('inf')

        for i in range(n_locations):
            if not visited[i] and current_location != i:
                input_data = np.zeros(2 * n_locations)
                input_data[current_location] = 1
                input_data[n_locations + i] = 1
                travel_time = model(torch.FloatTensor(input_data)).item()
                start_working_time, end_working_time = attractions_df.loc[i, 'Working Hours']
                arrival_time = current_time + travel_time

                if start_working_time <= arrival_time <= end_working_time and total_time + travel_time + time_per_location <= max_time:
                    if travel_time < min_time:
                        min_time = travel_time
                        next_location = i

        if next_location is None:
            break

        total_time += min_time
        total_distance += distance_matrix[current_location, next_location]
        current_time += min_time
        current_location = next_location

    return route, total_time, total_distance

# Прогноз маршрута
best_route, total_time, total_distance = predict_route(model, attractions_df, time_matrix, max_time, time_per_location)
total_places_visited = len(best_route)

print(f'Best Route: {best_route}')
print(f'Total Time: {total_time:.2f} hours, Total Distance: {total_distance:.2f} km')
print(f'Total Number of Visited Places: {total_places_visited}')

# Map route indices to attraction names
best_route_names = attractions_df.loc[best_route, 'Название'].tolist()
print(f'Best Route Names: {best_route_names}')
