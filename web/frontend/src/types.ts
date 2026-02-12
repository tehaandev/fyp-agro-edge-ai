export interface SensorFormData {
  humidity: number;
  atmospheric_temp: number;
  soil_temp: number;
  soil_moisture: number;
  dew_point: number;
  hour_of_day: number;
  day_of_week: number;
  nitrogen: number;
  phosphorus: number;
  potassium: number;
  soil_ph: number;
  rainfall: number;
}

export interface DiseaseResult {
  status: string;
  predicted_class: string | null;
  confidence: number;
  all_probabilities: Record<string, number>;
  confidence_level: string;
  is_fungal: boolean;
  is_viral: boolean;
  error: string | null;
}

export interface IrrigationResult {
  status: string;
  decision: string | null;
  original_decision: string | null;
  probabilities: Record<string, number>;
  confidence: number;
  disease_adjusted: boolean;
  adjustment_reason: string | null;
  error: string | null;
}

export interface CropResult {
  status: string;
  suitability_score: number;
  suitability_status: string | null;
  limiting_factors: string[];
  error: string | null;
}

export interface FinalAdvice {
  summary: string;
  urgency: string;
  actions: string[];
  warnings: string[];
  confidence_assessment: string;
}

export interface AnalysisResponse {
  timestamp: string;
  disease_detection: DiseaseResult;
  irrigation_decision: IrrigationResult;
  crop_recommendation: CropResult;
  final_advice: FinalAdvice;
  system_notes: {
    models_loaded: string[];
    execution_time_ms: number;
    warnings: string[];
  };
}
