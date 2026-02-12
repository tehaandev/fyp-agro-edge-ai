import { useState } from "react";
import type { SensorFormData, AnalysisResponse } from "./types";
import { runAnalysis } from "./api";
import ResultsDashboard from "./ResultsDashboard";

const DEFAULT_SENSOR_DATA: SensorFormData = {
  humidity: 72.5,
  atmospheric_temp: 28.3,
  soil_temp: 26.1,
  soil_moisture: 45.2,
  dew_point: 22.4,
  hour_of_day: new Date().getHours(),
  day_of_week: (new Date().getDay() + 6) % 7, // JS Sunday=0 -> Monday=0
  nitrogen: 120.0,
  phosphorus: 65.0,
  potassium: 155.0,
  soil_ph: 6.4,
  rainfall: 850.0,
};

const FIELD_CONFIG: {
  key: keyof SensorFormData;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  group: "env" | "soil";
}[] = [
  { key: "humidity", label: "Humidity", unit: "%", min: 0, max: 100, step: 0.1, group: "env" },
  { key: "atmospheric_temp", label: "Atmospheric Temp", unit: "°C", min: 0, max: 50, step: 0.1, group: "env" },
  { key: "soil_temp", label: "Soil Temp", unit: "°C", min: 0, max: 50, step: 0.1, group: "env" },
  { key: "soil_moisture", label: "Soil Moisture", unit: "", min: 0, max: 100, step: 0.1, group: "env" },
  { key: "dew_point", label: "Dew Point", unit: "°C", min: -10, max: 40, step: 0.1, group: "env" },
  { key: "hour_of_day", label: "Hour of Day", unit: "h", min: 0, max: 23, step: 1, group: "env" },
  { key: "day_of_week", label: "Day of Week (0=Mon)", unit: "", min: 0, max: 6, step: 1, group: "env" },
  { key: "nitrogen", label: "Nitrogen (N)", unit: "kg/ha", min: 0, max: 300, step: 1, group: "soil" },
  { key: "phosphorus", label: "Phosphorus (P)", unit: "kg/ha", min: 0, max: 200, step: 1, group: "soil" },
  { key: "potassium", label: "Potassium (K)", unit: "kg/ha", min: 0, max: 400, step: 1, group: "soil" },
  { key: "soil_ph", label: "Soil pH", unit: "", min: 3, max: 10, step: 0.1, group: "soil" },
  { key: "rainfall", label: "Annual Rainfall", unit: "mm", min: 0, max: 3000, step: 10, group: "soil" },
];

function App() {
  const [sensorData, setSensorData] = useState<SensorFormData>(DEFAULT_SENSOR_DATA);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (ev) => setImagePreview(ev.target?.result as string);
      reader.readAsDataURL(file);
    }
  };

  const handleFieldChange = (key: keyof SensorFormData, value: string) => {
    setSensorData((prev) => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!imageFile) {
      setError("Please upload a leaf image.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runAnalysis(imageFile, sensorData);
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const envFields = FIELD_CONFIG.filter((f) => f.group === "env");
  const soilFields = FIELD_CONFIG.filter((f) => f.group === "soil");

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-primary text-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center gap-3">
          <svg className="w-8 h-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 22c-4-3-8-6-8-11a8 8 0 0116 0c0 5-4 8-8 11z" />
            <path d="M12 11V6" />
            <path d="M9 8l3-3 3 3" />
          </svg>
          <div>
            <h1 className="text-xl font-bold">Agro Edge AI</h1>
            <p className="text-sm text-green-100">Smart Farm Decision Support</p>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Image Upload */}
          <section className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              Leaf Image
            </h2>
            <div className="flex items-start gap-6">
              <label className="flex-1 flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6 cursor-pointer hover:border-primary transition-colors">
                <svg className="w-10 h-10 text-gray-400 mb-2" fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                </svg>
                <span className="text-sm text-gray-500">
                  {imageFile ? imageFile.name : "Click to upload leaf image"}
                </span>
                <input
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={handleImageChange}
                />
              </label>
              {imagePreview && (
                <img
                  src={imagePreview}
                  alt="Preview"
                  className="w-32 h-32 object-cover rounded-lg border"
                />
              )}
            </div>
          </section>

          {/* Environmental Sensors */}
          <section className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              Environmental Sensors
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {envFields.map((f) => (
                <div key={f.key}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {f.label} {f.unit && <span className="text-gray-400">({f.unit})</span>}
                  </label>
                  <input
                    type="number"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={sensorData[f.key]}
                    onChange={(e) => handleFieldChange(f.key, e.target.value)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                  />
                </div>
              ))}
            </div>
          </section>

          {/* Soil Nutrients */}
          <section className="bg-white rounded-lg shadow p-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">
              Soil Nutrients
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {soilFields.map((f) => (
                <div key={f.key}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {f.label} {f.unit && <span className="text-gray-400">({f.unit})</span>}
                  </label>
                  <input
                    type="number"
                    min={f.min}
                    max={f.max}
                    step={f.step}
                    value={sensorData[f.key]}
                    onChange={(e) => handleFieldChange(f.key, e.target.value)}
                    className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
                  />
                </div>
              ))}
            </div>
          </section>

          {/* Submit */}
          <div className="flex items-center gap-4">
            <button
              type="submit"
              disabled={loading}
              className="bg-primary hover:bg-primary-dark text-white font-semibold px-8 py-3 rounded-lg shadow transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>
            {loading && (
              <span className="text-sm text-gray-500">
                Running 3 ML models, please wait...
              </span>
            )}
          </div>

          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
              {error}
            </div>
          )}
        </form>

        {/* Results */}
        {result && <ResultsDashboard data={result} />}
      </main>
    </div>
  );
}

export default App;
