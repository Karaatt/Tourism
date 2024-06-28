document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    form.addEventListener("submit", function() {
        const routeSection = document.querySelector(".route-section");
        routeSection.style.opacity = 0;
        setTimeout(() => {
            routeSection.style.opacity = 1;
        }, 500);
    });

    ymaps.ready(initMap);
});

function initMap() {
    const mapContainer = document.getElementById("map");
    const map = new ymaps.Map(mapContainer, {
        center: [45.035470, 38.975313], // Координаты центра карты (Ставрополь, например)
        zoom: 12, // Уровень масштабирования карты
        controls: ['zoomControl', 'fullscreenControl'] // Добавляем элементы управления
    });

    // Пример создания маркера
    const myPlacemark = new ymaps.Placemark([45.035470, 38.975313], {
        hintContent: 'Start Point',
        balloonContent: 'Your start point is here!'
    });

    // Добавление маркера на карту
    map.geoObjects.add(myPlacemark);

    // Функция для построения маршрута
    function buildRoute(startPoint, endPoint) {
        ymaps.route([
            startPoint,
            endPoint
        ]).then(function (route) {
            // Добавление маршрута на карту
            map.geoObjects.add(route);
        }, function (error) {
            console.error('Error building route:', error);
        });
    }

    // Пример использования функции для построения маршрута
    // Здесь startPoint и endPoint должны быть координатами точек маршрута
    const startPoint = [45.035470, 38.975313];
    const endPoint = [45.050000, 39.000000];
    buildRoute(startPoint, endPoint);
}
