
document.addEventListener("DOMContentLoaded", function () {
    const sidebar = document.getElementById("sidebar");
    const menuToggle = document.getElementById("menuToggle");
    const navLinks = document.querySelectorAll(".sidebar-nav .nav-link");

    // Open or close the sidebar on smaller screens.
    if (menuToggle && sidebar) {
        menuToggle.addEventListener("click", function () {
            sidebar.classList.toggle("open");
        });
    }

    // Highlight the clicked navigation item.
    navLinks.forEach(function (link) {
        link.addEventListener("click", function () {
            navLinks.forEach(function (item) {
                item.classList.remove("active");
            });

            this.classList.add("active");
        });
    });

    // Close the mobile sidebar if the user clicks outside it.
    document.addEventListener("click", function (event) {
        if (
            sidebar &&
            menuToggle &&
            window.innerWidth <= 850 &&
            sidebar.classList.contains("open") &&
            !sidebar.contains(event.target) &&
            !menuToggle.contains(event.target)
        ) {
            sidebar.classList.remove("open");
        }
    });
});
