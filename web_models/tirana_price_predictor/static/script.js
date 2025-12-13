// static/script.js
document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("predict-form");
    const resultDiv = document.getElementById("result");

    const priceEl = document.getElementById("metric-price");
    const rangeEl = document.getElementById("metric-range");
    const ppm2El = document.getElementById("metric-ppm2");
    const commentEl = document.getElementById("result-comment");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Reset UI state
        resultDiv.style.display = "none";
        resultDiv.classList.remove("error");
        resultDiv.classList.remove("result-placeholder");
        resultDiv.textContent = "Calculating...";

        if (priceEl) priceEl.textContent = "—";
        if (rangeEl) rangeEl.textContent = "—";
        if (ppm2El) ppm2El.textContent = "—";
        if (commentEl) {
            commentEl.textContent =
                "The estimate will appear here together with an indicative range once you submit the form.";
        }

        const payload = {
            area_m2: Number(document.getElementById("area_m2").value),
            bathrooms: Number(document.getElementById("bathrooms").value),
            bedrooms: Number(document.getElementById("bedrooms").value),
            floor: Number(document.getElementById("floor").value),
            furnishing: document.getElementById("furnishing").value,
            dist_to_blloku_km: Number(document.getElementById("dist_to_blloku_km").value),
            has_elevator: document.getElementById("has_elevator").checked,
            has_parking: document.getElementById("has_parking").checked,
            has_garage: document.getElementById("has_garage").checked,
            has_terrace: document.getElementById("has_terrace").checked,
        };

        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                throw new Error("Server error");
            }

            const data = await response.json();
            const price = data.predicted_price_eur;
            const [low, high] =
                data.confidence_interval || [Math.round(price * 0.75), Math.round(price * 1.25)];

            const area = payload.area_m2 || 0;
            const ppm2 = area > 0 ? Math.round(price / area) : null;

            // Main text block
            resultDiv.classList.remove("error");
            resultDiv.innerHTML = `
                Estimated price: <strong>${price.toLocaleString("en-US")} €</strong><br>
                Approximate range: ${low.toLocaleString("en-US")} € – ${high.toLocaleString("en-US")} €
            `;

            // Metric tiles on right
            if (priceEl) {
                priceEl.textContent = `${price.toLocaleString("en-US")} €`;
            }
            if (rangeEl) {
                rangeEl.textContent = `${low.toLocaleString("en-US")} € – ${high.toLocaleString("en-US")} €`;
            }
            if (ppm2El) {
                ppm2El.textContent = ppm2 ? `${ppm2.toLocaleString("en-US")} €/m²` : "—";
            }
            if (commentEl) {
                commentEl.textContent =
                    "This estimate is based on size, configuration, distance to Blloku and selected amenities.";
            }
        } catch (err) {
            resultDiv.classList.add("error");
            resultDiv.textContent =
                "An error occurred while predicting the price. Please try again.";

            if (commentEl) {
                commentEl.textContent =
                    "The service is temporarily unavailable. Please retry in a moment.";
            }
        }

        resultDiv.style.display = "block";
    });
});
