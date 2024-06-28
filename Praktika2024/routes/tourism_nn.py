import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Загрузка данных
attractions_df = pd.read_excel('stavropol_attractions_coords__kopia_1.xlsx')
distance_df = pd.read_excel('distance.xlsx', index_col=0)

# Конвертирование расстояний из метров в километры
distance_matrix = distance_df.values / 1000

# Параметры
time_per_location = 20 / 60  # 20 минут в часах
max_time = 24  # 24 часа
speed = 40  # 40 км/ч

# Конвертирование расстояний во время (в часах)
time_matrix = distance_matrix / speed

# Функция для парсинга времени работы
def parse_working_hours(time_str):
    try:
        start_time, end_time = time_str.split('–')
        start_hour, start_minute = map(int, start_time.split(':'))
        end_hour, end_minute = map(int, end_time.split(':'))
        start_time_in_hours = start_hour + start_minute / 60
        end_time_in_hours = end_hour + end_minute / 60
        return start_time_in_hours, end_time_in_hours
    except ValueError:
        return 0, 24  # Открыто весь день, если парсинг не удался

# Добавление рабочего времени в датафрейм достопримечательностей
attractions_df['Working Hours'] = attractions_df['Время работы'].apply(parse_working_hours)

# Сортировка достопримечательностей по рейтингу (меньший рейтинг - выше приоритет)
attractions_df = attractions_df.sort_values(by='Рейтинг').reset_index()

class RouteOptimizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RouteOptimizer, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x.view(1, -1, len(x)), hidden)
        out = self.fc(out.view(-1, self.hidden_size))
        return out, hidden

    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_size), torch.zeros(2, 1, self.hidden_size))

# Подготовка данных
n_locations = time_matrix.shape[0]
input_size = n_locations
hidden_size = 256
output_size = n_locations

model = RouteOptimizer(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Конвертирование матрицы времени в тензор
time_matrix_tensor = torch.tensor(time_matrix, dtype=torch.float32)

def train(model, criterion, optimizer, time_matrix_tensor, n_epochs=1000):
    for epoch in range(n_epochs):
        hidden = model.init_hidden()
        optimizer.zero_grad()

        # Подготовка входных и целевых тензоров
        inputs = time_matrix_tensor
        targets = torch.arange(n_locations, dtype=torch.float32)

        # Прямой проход
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs, targets)

        # Обратный проход и оптимизация
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')

def predict_route(start_point, time_limit):
    # Проверка существования стартовой локации
    if start_point not in attractions_df['Название'].values:
        raise IndexError("Start point not found in attractions")

    model.eval()
    hidden = model.init_hidden()
    inputs = time_matrix_tensor
    outputs, hidden = model(inputs, hidden)
    _, predicted_route = torch.topk(outputs, n_locations)

    # Учитывание рабочего времени и приоритизация по рейтингу
    final_route = []
    visited = [False] * n_locations
    current_time = 0
    total_time = 0
    total_distance = 0

    # Определение стартовой точки
    start_idx = attractions_df[attractions_df['Название'] == start_point].index[0]

    while total_time + time_per_location <= time_limit:
        next_location = None
        min_time = float('inf')

        for loc in predicted_route[0].tolist():
            if not visited[loc]:
                start_working_time, end_working_time = attractions_df.loc[loc, 'Working Hours']
                if current_time < start_working_time and start_working_time != 0:
                    continue  # Пропустить, чтобы найти места, которые открыты

                travel_time = 0 if not final_route else time_matrix[final_route[-1], loc]
                arrival_time = current_time + travel_time

                if start_working_time <= arrival_time <= end_working_time and total_time + travel_time + time_per_location <= time_limit:
                    next_location = loc
                    min_time = travel_time
                    break

        if next_location is None:
            # Если не найдено ни одного подходящего места, проверка мест 24/7
            for loc in predicted_route[0].tolist():
                if not visited[loc]:
                    start_working_time, end_working_time = attractions_df.loc[loc, 'Working Hours']
                    if start_working_time == 0 and end_working_time == 24:
                        next_location = loc
                        min_time = time_matrix[final_route[-1], loc] if final_route else 0
                        break

            if next_location is None:
                break

        final_route.append(next_location)
        visited[next_location] = True
        total_time += time_per_location + min_time
        total_distance += min_time * speed
        current_time += time_per_location + min_time

    # Преобразование индексов маршрута в имена достопримечательностей
    predicted_route_names = attractions_df.loc[final_route, 'Название'].tolist()

    return predicted_route_names, total_time, total_distance

# Тренировка модели (необязательно, если модель уже обучена и сохранена)
#train(model, criterion, optimizer, time_matrix_tensor)

# Сохранение и загрузка модели (если необходимо)
torch.save(model.state_dict(), 'route_optimizer.pth')
model.load_state_dict(torch.load('route_optimizer.pth'))
