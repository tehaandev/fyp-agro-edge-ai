import type { AnalysisResponse } from "./types";

const URGENCY_STYLES: Record<string, string> = {
  low: "bg-green-50 border-green-300 text-green-800",
  normal: "bg-blue-50 border-blue-300 text-blue-800",
  high: "bg-orange-50 border-orange-300 text-orange-800",
  critical: "bg-red-50 border-red-300 text-red-800",
};

const URGENCY_BADGE: Record<string, string> = {
  low: "bg-green-200 text-green-800",
  normal: "bg-blue-200 text-blue-800",
  high: "bg-orange-200 text-orange-800",
  critical: "bg-red-200 text-red-800",
};

function DiseaseCard({ data }: { data: AnalysisResponse["disease_detection"] }) {
  if (data.status !== "success") {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="font-semibold text-gray-800 mb-2">Disease Detection</h3>
        <p className="text-red-600 text-sm">Error: {data.error}</p>
      </div>
    );
  }

  const isHealthy = data.predicted_class === "Healthy";
  const borderColor = isHealthy ? "border-green-400" : "border-red-400";
  const bgColor = isHealthy ? "bg-green-50" : "bg-red-50";

  return (
    <div className={`bg-white rounded-lg shadow p-6 border-l-4 ${borderColor}`}>
      <h3 className="font-semibold text-gray-800 mb-3">Disease Detection</h3>
      <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-3 ${bgColor} ${isHealthy ? "text-green-800" : "text-red-800"}`}>
        {data.predicted_class}
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-500">Confidence</span>
          <span className="font-medium">{((data.confidence) * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className={`h-2 rounded-full ${isHealthy ? "bg-green-500" : "bg-red-500"}`}
            style={{ width: `${data.confidence * 100}%` }}
          />
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Level</span>
          <span className="capitalize">{data.confidence_level}</span>
        </div>
        {data.is_fungal && (
          <span className="inline-block px-2 py-0.5 bg-yellow-100 text-yellow-800 text-xs rounded">Fungal</span>
        )}
        {data.is_viral && (
          <span className="inline-block px-2 py-0.5 bg-purple-100 text-purple-800 text-xs rounded">Viral</span>
        )}
      </div>
      {/* Probability breakdown */}
      <details className="mt-3">
        <summary className="text-xs text-gray-400 cursor-pointer">All probabilities</summary>
        <div className="mt-2 space-y-1">
          {Object.entries(data.all_probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, prob]) => (
              <div key={cls} className="flex items-center gap-2 text-xs">
                <span className="w-36 text-gray-600 truncate">{cls}</span>
                <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                  <div className="bg-gray-400 h-1.5 rounded-full" style={{ width: `${prob * 100}%` }} />
                </div>
                <span className="w-12 text-right text-gray-500">{(prob * 100).toFixed(1)}%</span>
              </div>
            ))}
        </div>
      </details>
    </div>
  );
}

function IrrigationCard({ data }: { data: AnalysisResponse["irrigation_decision"] }) {
  if (data.status !== "success") {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="font-semibold text-gray-800 mb-2">Irrigation Control</h3>
        <p className="text-red-600 text-sm">Error: {data.error}</p>
      </div>
    );
  }

  const decisionColors: Record<string, string> = {
    Irrigate_High: "bg-blue-100 text-blue-800 border-blue-400",
    Irrigate_Low: "bg-cyan-100 text-cyan-800 border-cyan-400",
    No_Irrigation: "bg-gray-100 text-gray-800 border-gray-400",
  };
  const style = decisionColors[data.decision || ""] || decisionColors.No_Irrigation;

  return (
    <div className={`bg-white rounded-lg shadow p-6 border-l-4 ${style.split(" ").pop()}`}>
      <h3 className="font-semibold text-gray-800 mb-3">Irrigation Control</h3>
      <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-3 ${style}`}>
        {data.decision?.replace(/_/g, " ")}
      </div>
      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-500">Confidence</span>
          <span className="font-medium">{(data.confidence * 100).toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div className="h-2 rounded-full bg-blue-500" style={{ width: `${data.confidence * 100}%` }} />
        </div>
        {data.disease_adjusted && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-800 text-xs p-2 rounded mt-2">
            <span className="font-medium">Adjusted:</span> {data.adjustment_reason}
          </div>
        )}
      </div>
      {/* Probabilities */}
      <div className="mt-3 space-y-1">
        {Object.entries(data.probabilities)
          .sort(([, a], [, b]) => b - a)
          .map(([cls, prob]) => (
            <div key={cls} className="flex items-center gap-2 text-xs">
              <span className="w-28 text-gray-600">{cls.replace(/_/g, " ")}</span>
              <div className="flex-1 bg-gray-100 rounded-full h-1.5">
                <div className="bg-blue-400 h-1.5 rounded-full" style={{ width: `${prob * 100}%` }} />
              </div>
              <span className="w-12 text-right text-gray-500">{(prob * 100).toFixed(1)}%</span>
            </div>
          ))}
      </div>
    </div>
  );
}

function SuitabilityCard({ data }: { data: AnalysisResponse["crop_recommendation"] }) {
  if (data.status !== "success") {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="font-semibold text-gray-800 mb-2">Crop Suitability</h3>
        <p className="text-red-600 text-sm">Error: {data.error}</p>
      </div>
    );
  }

  const statusColors: Record<string, string> = {
    Suitable: "text-green-700 border-green-400",
    Marginal: "text-yellow-700 border-yellow-400",
    Not_Suitable: "text-red-700 border-red-400",
  };
  const scoreColor =
    data.suitability_score >= 65 ? "bg-green-500" : data.suitability_score >= 40 ? "bg-yellow-500" : "bg-red-500";
  const style = statusColors[data.suitability_status || ""] || statusColors.Marginal;

  return (
    <div className={`bg-white rounded-lg shadow p-6 border-l-4 ${style.split(" ").pop()}`}>
      <h3 className="font-semibold text-gray-800 mb-3">Crop Suitability</h3>
      <div className="flex items-end gap-2 mb-3">
        <span className={`text-3xl font-bold ${style.split(" ")[0]}`}>
          {data.suitability_score.toFixed(0)}
        </span>
        <span className="text-gray-500 text-sm mb-1">/ 100</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3 mb-3">
        <div className={`h-3 rounded-full ${scoreColor}`} style={{ width: `${data.suitability_score}%` }} />
      </div>
      <span className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${data.suitability_status === "Suitable" ? "bg-green-100 text-green-800" : data.suitability_status === "Marginal" ? "bg-yellow-100 text-yellow-800" : "bg-red-100 text-red-800"}`}>
        {data.suitability_status?.replace(/_/g, " ")}
      </span>
      {data.limiting_factors.length > 0 && (
        <div className="mt-3">
          <p className="text-xs font-medium text-gray-500 mb-1">Limiting Factors:</p>
          <ul className="space-y-1">
            {data.limiting_factors.map((f, i) => (
              <li key={i} className="text-xs text-orange-700 bg-orange-50 px-2 py-1 rounded">
                {f}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default function ResultsDashboard({ data }: { data: AnalysisResponse }) {
  const urgency = data.final_advice.urgency || "normal";

  return (
    <div className="mt-8 space-y-6">
      {/* Final Advice Banner */}
      <div className={`border rounded-lg p-5 ${URGENCY_STYLES[urgency]}`}>
        <div className="flex items-center gap-3 mb-2">
          <span className={`text-xs font-bold px-2 py-0.5 rounded uppercase ${URGENCY_BADGE[urgency]}`}>
            {urgency}
          </span>
          <h2 className="font-bold text-lg">{data.final_advice.summary}</h2>
        </div>
        <p className="text-xs opacity-70 mb-3">{data.final_advice.confidence_assessment}</p>

        {data.final_advice.actions.length > 0 && (
          <div className="mb-3">
            <p className="text-sm font-semibold mb-1">Actions:</p>
            <ul className="space-y-1">
              {data.final_advice.actions.map((a, i) => (
                <li key={i} className="text-sm">{a}</li>
              ))}
            </ul>
          </div>
        )}

        {data.final_advice.warnings.length > 0 && (
          <div>
            <p className="text-sm font-semibold mb-1">Warnings:</p>
            <ul className="space-y-1">
              {data.final_advice.warnings.map((w, i) => (
                <li key={i} className="text-sm font-medium">{w}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Model Result Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <DiseaseCard data={data.disease_detection} />
        <IrrigationCard data={data.irrigation_decision} />
        <SuitabilityCard data={data.crop_recommendation} />
      </div>

      {/* System Info */}
      <div className="text-xs text-gray-400 flex gap-4">
        <span>Models loaded: {data.system_notes.models_loaded.join(", ") || "none"}</span>
        <span>Execution: {data.system_notes.execution_time_ms}ms</span>
        <span>Timestamp: {new Date(data.timestamp).toLocaleString()}</span>
      </div>
    </div>
  );
}
