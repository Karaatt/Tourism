// Функция для инициализации карты
function initMap() {
    var map = new ymaps.Map("map", {
        center: [45.042, 38.973], // Начальные координаты (можно адаптировать)
        zoom: 12 // Уровень масштабирования карты
    });

    // Добавление маркеров достопримечательностей
    {% for attraction in attractions %}
        var placemark{{ attraction.id }} = new ymaps.Placemark(
            [{{ attraction.latitude }}, {{ attraction.longitude }}],
            {
                hintContent: "{{ attraction.name }}",
                balloonContent: "{{ attraction.name }}"
            },
            {
                preset: 'islands#blueDotIcon' // Стиль маркера
            }
        );

        map.geoObjects.add(placemark{{ attraction.id }});
    {% endfor %}
}

// Загрузка карты после полной загрузки страницы
document.addEventListener('DOMContentLoaded', function () {
    ymaps.ready(initMap);
});

// Функция для отправки формы с выбором точек маршрута
function planRoute() {
    var form = document.getElementById('routeForm');
    form.submit();
}
