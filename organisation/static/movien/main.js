var cw = window.average.clientWidth;
var avg = document.getElementById('average').getAttribute('value');
// var star = document.getElementById('noo').getAttribute('value');

function rating( stars ) {
  window.average.style.width = Math.round(cw * (stars / 5)) + 'px';
}

rating(avg);
// rating(star);

