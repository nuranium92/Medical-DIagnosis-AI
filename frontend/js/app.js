(function () {
    const tabBtns     = document.querySelectorAll(".tab-btn");
    const tabSections = document.querySelectorAll(".tab-section");

    tabBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;

            tabBtns.forEach((b) => b.classList.remove("active"));
            tabSections.forEach((s) => s.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(`tab-${target}`).classList.add("active");
        });
    });
})();