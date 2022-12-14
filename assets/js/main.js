$( document ).ready(function() {
    if(window.location.pathname !== '/') {
        $('.navbar-brand').removeClass("invisible");
    }
});