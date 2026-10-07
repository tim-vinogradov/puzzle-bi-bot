/* Русская локаль Plotly: подписи панели инструментов, месяцы, разделители чисел */
(function () {
  if (typeof Plotly === 'undefined') return;
  Plotly.register({
    moduleType: 'locale',
    name: 'ru',
    dictionary: {
      'Autoscale': 'Автомасштаб',
      'Box Select': 'Выделение прямоугольником',
      'Click to enter Colorscale title': 'Нажмите, чтобы ввести название шкалы',
      'Double-click to zoom back out': 'Двойной клик — вернуть масштаб',
      'Download plot': 'Скачать график',
      'Download plot as a png': 'Скачать график (PNG)',
      'Lasso Select': 'Выделение лассо',
      'Orbital rotation': 'Вращение по орбите',
      'Pan': 'Сдвиг',
      'Produced with Plotly.js': 'Создано в Plotly.js',
      'Reset': 'Сбросить',
      'Reset axes': 'Сбросить оси',
      'Reset view': 'Сбросить вид',
      'Show closest data on hover': 'Подсказка у ближайшей точки',
      'Snapshot succeeded': 'Снимок готов',
      'Sorry, there was a problem downloading your snapshot!': 'Не удалось скачать снимок',
      'Taking snapshot - this may take a few seconds': 'Делаю снимок — это займёт пару секунд',
      'Toggle Spike Lines': 'Направляющие линии',
      'Toggle show closest data on hover': 'Переключить подсказки',
      'Zoom': 'Масштаб',
      'Zoom in': 'Приблизить',
      'Zoom out': 'Отдалить',
      'Compare data on hover': 'Сравнить значения при наведении',
      'trace': 'ряд',
      'max': 'макс.',
      'min': 'мин.',
      'mean': 'среднее',
      'median': 'медиана'
    },
    format: {
      days: ['Воскресенье', 'Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота'],
      shortDays: ['Вс', 'Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб'],
      months: ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь'],
      shortMonths: ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек'],
      date: '%d.%m.%Y',
      decimal: ',',
      thousands: ' '
    }
  });
})();
