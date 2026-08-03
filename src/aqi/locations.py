"""India AQI locations: 28 states + 8 union territories."""

INDIA_CITIES = {
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "search": "visakhapatnam"},
    "Itanagar": {"lat": 27.0844, "lon": 93.6053, "state": "Arunachal Pradesh", "search": "itanagar"},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam", "search": "guwahati"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar", "search": "patna"},
    "Raipur": {"lat": 21.2514, "lon": 81.6296, "state": "Chhattisgarh", "search": "raipur"},
    "Panaji": {"lat": 15.4909, "lon": 73.8278, "state": "Goa", "search": "goa"},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "search": "ahmedabad"},
    "Faridabad": {"lat": 28.4089, "lon": 77.3178, "state": "Haryana", "search": "faridabad"},
    "Shimla": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh", "search": "shimla"},
    "Ranchi": {"lat": 23.3441, "lon": 85.3096, "state": "Jharkhand", "search": "ranchi"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "search": "bangalore"},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala", "search": "thiruvananthapuram"},
    "Indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh", "search": "indore"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "search": "mumbai"},
    "Imphal": {"lat": 24.8170, "lon": 93.9368, "state": "Manipur", "search": "imphal"},
    "Shillong": {"lat": 25.5788, "lon": 91.8933, "state": "Meghalaya", "search": "shillong"},
    "Aizawl": {"lat": 23.7307, "lon": 92.7173, "state": "Mizoram", "search": "aizawl"},
    "Kohima": {"lat": 25.6747, "lon": 94.1086, "state": "Nagaland", "search": "kohima"},
    "Bhubaneswar": {"lat": 20.2961, "lon": 85.8245, "state": "Odisha", "search": "bhubaneswar"},
    "Ludhiana": {"lat": 30.9010, "lon": 75.8573, "state": "Punjab", "search": "ludhiana"},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "search": "jaipur"},
    "Gangtok": {"lat": 27.3389, "lon": 88.6065, "state": "Sikkim", "search": "gangtok"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "search": "chennai"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana", "search": "hyderabad"},
    "Agartala": {"lat": 23.8315, "lon": 91.2868, "state": "Tripura", "search": "agartala"},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "search": "lucknow"},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand", "search": "dehradun"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "search": "kolkata"},
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "Delhi (NCT)", "search": "delhi"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Chandigarh (UT)", "search": "chandigarh"},
    "Puducherry": {"lat": 11.9416, "lon": 79.8083, "state": "Puducherry (UT)", "search": "puducherry"},
    "Srinagar": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir (UT)", "search": "srinagar"},
    "Leh": {"lat": 34.1526, "lon": 77.5771, "state": "Ladakh (UT)", "search": "leh"},
    "Port_Blair": {"lat": 11.6234, "lon": 92.7265, "state": "Andaman & Nicobar (UT)", "search": "port blair"},
    "Daman": {"lat": 20.4283, "lon": 72.8397, "state": "Daman & Diu (UT)", "search": "daman"},
    "Kavaratti": {"lat": 10.5667, "lon": 72.6417, "state": "Lakshadweep (UT)", "search": "kavaratti"},
}

INDIAN_HOLIDAYS = [
    *[f"{y}-01-26" for y in range(2015, 2027)],
    *[f"{y}-08-15" for y in range(2015, 2027)],
    *[f"{y}-10-02" for y in range(2015, 2027)],
    "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08",  # Diwali
    "2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04",  # Holi
]
