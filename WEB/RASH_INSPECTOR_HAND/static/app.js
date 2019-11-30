const navslide = () => {
    const burger = document.querySelector('.burger');
    const nav = document.querySelector('.nav-links');
    const navlinks = document.querySelectorAll('.nav-links li');



    burger.addEventListener('click', () => {

        nav.classList.toggle('nav-active');

        navlinks.forEach((link, index) => {
            if (link.style.animation) {
                link.style.animation = ''
            } else {
                link.style.animation = `navlinkfade 0.5s ease forwards ${index / 7 + 0.5}s`;
                console.log(index / 7 + 0.5);
            }
        });

        burger.classList.toggle('toggle');

    });


}

navslide();