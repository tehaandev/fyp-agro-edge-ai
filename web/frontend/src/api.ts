import type { SensorFormData, AnalysisResponse } from "./types";

export async function runAnalysis(
  image: File,
  sensorData: SensorFormData
): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("image", image);

  for (const [key, value] of Object.entries(sensorData)) {
    formData.append(key, String(value));
  }

  const res = await fetch("/api/analyze", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Analysis failed (${res.status}): ${text}`);
  }

  return res.json();
}
