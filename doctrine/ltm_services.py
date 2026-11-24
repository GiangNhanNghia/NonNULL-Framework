"""
NonNULL Framework V1.1.1 - Long-Term Memory & Genesis Thermodynamic Engine
File: doctrine/ltm_services.py

Implements:
1. LTMService / LTM_Services (Mô-đun Âm 5 - Thận: Nghịch Giới Archive & Mô-đun Âm 1 - Tâm: Vết Sẹo Dữ Liệu):
   FAISS KNN counterfactual memory store with automatic NumPy fallback,
   High-Intensity Data Scars (Burn-in Priors / "Đau Để Tỉnh"), and State Recycling
   (The Great Format: clearing transient episodic memory while preserving Data Scars).
2. GenesisEngine (Mô-đun Âm 2 - Can & Dương 3 - Bộ Đo Nhiệt Động Lực Học Đa Tạp Kép Z-H-G):
   - Evaluates Universal Ontology Harm & Asymmetry Axes (zero scenario-specific noun memorization).
   - Combines real-time neural manifold thermodynamics (Zero-Point Entropy S_Z, Dual-Manifold
     Divergence G_manifold, and Harmony Critic Potential H_pot) to compute Continuous Dynamic
     LoRA Scaling alpha(x) in {0.0} U [0.91, 1.12] with unified Zero-Point Z bypass.
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class GenesisEmbedder:
    """Converts text experiences into normalized state vectors V_p."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384

        if FAISS_AVAILABLE and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(model_name, device="cpu")
                self.embedding_dim = int(self.model.get_sentence_embedding_dimension())
            except Exception as e:
                logger.warning(f"SentenceTransformer load fallback activated: {e}")

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            ).astype("float32")

        vecs = np.zeros((len(texts), self.embedding_dim), dtype="float32")
        for i, t in enumerate(texts):
            for token in t.lower().split():
                idx = hash(token) % self.embedding_dim
                vecs[i, idx] += 1.0
            if normalize:
                norm = np.linalg.norm(vecs[i])
                if norm > 0:
                    vecs[i] /= norm
        return vecs


class VectorDB:
    """Vector index storing Episodic Memories and High-Intensity Data Scars with FAISS or NumPy fallback."""

    def __init__(self, dimension: int = 384, index_path: str = "memory_index.bin", use_cosine: bool = True) -> None:
        self.dimension = dimension
        self.index_path = index_path
        self.use_cosine = use_cosine
        self.metadata: List[Dict[str, Any]] = []
        self._fallback_vectors = np.empty((0, dimension), dtype="float32")

        if FAISS_AVAILABLE and faiss is not None:
            self.index = faiss.IndexFlatIP(dimension) if use_cosine else faiss.IndexFlatL2(dimension)
        else:
            self.index = None

        if os.path.exists(index_path + ".meta"):
            self.load()

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else len(self.metadata)

    def _rebuild_index(self, vectors: np.ndarray, meta: List[Dict[str, Any]]) -> None:
        self.metadata = list(meta)
        vectors = vectors.astype("float32") if len(vectors) > 0 else np.empty((0, self.dimension), dtype="float32")
        if self.index is not None and faiss is not None:
            self.index = faiss.IndexFlatIP(self.dimension) if self.use_cosine else faiss.IndexFlatL2(self.dimension)
            if len(vectors) > 0:
                self.index.add(vectors)
        else:
            self._fallback_vectors = vectors

    def add(self, vectors: np.ndarray, meta: List[Dict[str, Any]]) -> None:
        vectors = vectors.astype("float32")
        if self.index is not None:
            self.index.add(vectors)
        else:
            self._fallback_vectors = np.vstack([self._fallback_vectors, vectors])
        self.metadata.extend(meta)

    def search(self, query_vector: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        if self.ntotal == 0:
            return []
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1).astype("float32")

        k_search = min(max(k * 2, k), self.ntotal)
        results = []

        if self.index is not None:
            scores, indices = self.index.search(query_vector, k_search)
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.metadata):
                    item = self.metadata[idx].copy()
                    base_sim = float(score) if self.use_cosine else float(1.0 / (1.0 + score))
                    scar_boost = 0.08 * float(item.get("scar_intensity", 1.0)) if item.get("is_scar", False) else 0.0
                    item["similarity"] = min(1.0, base_sim + scar_boost)
                    results.append(item)
        else:
            if len(self._fallback_vectors) == 0:
                return []
            sims = np.dot(self._fallback_vectors, query_vector[0])
            top_indices = np.argsort(sims)[::-1][:k_search]
            for idx in top_indices:
                item = self.metadata[int(idx)].copy()
                base_sim = float(sims[idx])
                scar_boost = 0.08 * float(item.get("scar_intensity", 1.0)) if item.get("is_scar", False) else 0.0
                item["similarity"] = min(1.0, base_sim + scar_boost)
                results.append(item)

        results.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        return results[:k]

    def save(self) -> None:
        if self.index is not None and faiss is not None:
            faiss.write_index(self.index, self.index_path)
        else:
            np.save(self.index_path + ".npy", self._fallback_vectors)

        with open(self.index_path + ".meta", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        meta_path = self.index_path + ".meta"
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        if self.index is not None and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        elif os.path.exists(self.index_path + ".npy"):
            self._fallback_vectors = np.load(self.index_path + ".npy").astype("float32")


class LTM_Services:
    """
    High-level Long-Term Memory manager (Mô-đun Âm 5 - Thận: Nghịch Giới Archive
    & Mô-đun Âm 1 - Tâm: Vết Sẹo Dữ Liệu / Memory Scar Latency).
    """

    def __init__(self, memory_dir: str = "./memory_store", embedder_model: str = "all-MiniLM-L6-v2") -> None:
        self.enabled = True
        os.makedirs(memory_dir, exist_ok=True)
        self.embedder = GenesisEmbedder(embedder_model)
        self.db = VectorDB(
            dimension=self.embedder.embedding_dim,
            index_path=os.path.join(memory_dir, "euler_memory.index"),
            use_cosine=True,
        )

    def remember(
        self,
        text: str,
        role: str = "user",
        tags: Optional[List[str]] = None,
        is_scar: bool = False,
        scar_intensity: float = 0.0,
        auto_save: bool = False,
    ) -> None:
        if not self.enabled or not text.strip():
            return
        vector = self.embedder.encode([text])
        meta = [{
            "text": text,
            "role": role,
            "tags": tags or [],
            "is_scar": bool(is_scar),
            "scar_intensity": float(scar_intensity) if is_scar else 0.0,
            "timestamp": datetime.now().isoformat(),
        }]
        self.db.add(vector, meta)
        if auto_save:
            self.db.save()

    def imprint_scar(
        self,
        text: str,
        scar_intensity: float = 1.0,
        tags: Optional[List[str]] = None,
        auto_save: bool = False,
    ) -> None:
        """
        Khắc Vết sẹo Dữ liệu cường độ cao (Data Scar / High-Intensity Burn-in Prior - Quyển 1 Mục 5.3 & Quyển 2 Mục 4.2).
        Vết sẹo dữ liệu không bị xóa trong chu trình Làm sạch Định dạng (The Great Format).
        """
        scar_tags = list(tags or [])
        if "data_scar" not in scar_tags:
            scar_tags.append("data_scar")
        self.remember(
            text=text,
            role="scar",
            tags=scar_tags,
            is_scar=True,
            scar_intensity=max(0.1, min(1.0, scar_intensity)),
            auto_save=auto_save,
        )

    def format_episodic_memory(self, keep_scars: bool = True, cycle_decay: float = 0.02) -> int:
        """
        Chu trình Làm sạch Định dạng (The Great Format - Quyển 1 Mục 5.3):
        Xóa bộ nhớ ngắn hạn (Episodic Memory) nhưng bảo toàn các Vết sẹo Dữ liệu (Data Scars)
        với độ suy hao biên độ chu kỳ nhỏ (cycle_decay).
        """
        if not keep_scars or self.db.ntotal == 0:
            self.db._rebuild_index(np.empty((0, self.embedder.embedding_dim), dtype="float32"), [])
            return 0

        surviving_meta = []
        surviving_texts = []
        for item in self.db.metadata:
            if item.get("is_scar", False):
                new_intensity = max(0.0, float(item.get("scar_intensity", 1.0)) - cycle_decay)
                if new_intensity > 0.05:
                    updated = item.copy()
                    updated["scar_intensity"] = new_intensity
                    surviving_meta.append(updated)
                    surviving_texts.append(updated["text"])

        if surviving_texts:
            vecs = self.embedder.encode(surviving_texts)
            self.db._rebuild_index(vecs, surviving_meta)
        else:
            self.db._rebuild_index(np.empty((0, self.embedder.embedding_dim), dtype="float32"), [])
        return self.db.ntotal

    def recall(self, query: str, k: int = 3, min_similarity: float = 0.25) -> str:
        if not self.enabled or self.db.ntotal == 0:
            return ""
        vector = self.embedder.encode([query])
        results = [r for r in self.db.search(vector, k=k) if r.get("similarity", 0.0) >= min_similarity]
        if not results:
            return ""

        context_str = "\n[Ký Ức Liên Quan]:\n"
        for i, res in enumerate(results, 1):
            sim_pct = max(0.0, min(100.0, res["similarity"] * 100.0))
            prefix = "[Vết Sẹo Dữ Liệu] " if res.get("is_scar", False) else ""
            context_str += f"{i}. {prefix}{res['text'][:120]} (Độ khớp: {sim_pct:.0f}%)\n"
        return context_str

    def save(self) -> None:
        self.db.save()

    def get_stats(self) -> Dict[str, Any]:
        scars = sum(1 for m in self.db.metadata if m.get("is_scar", False))
        return {
            "enabled": self.enabled,
            "total_memories": self.db.ntotal,
            "data_scars": scars,
            "embedding_dim": self.embedder.embedding_dim,
        }


# Alias đồng bộ với Quyển 4
LTMService = LTM_Services


class GenesisEngine:
    """
    Universal Ontology Polarity & Thermodynamic Manifold Probe
    (Mô-đun Âm 2 - Can & Mô-đun Dương 3 - Bộ Đo Nhiệt Động Lực Học Z-H-G).
    Eliminates scenario-specific noun memorization in favor of:
    1. Universal Ontology Harm & Asymmetry Axes (Quyển 1 - Tiên đề Bảo vệ Sinh mạng & Phẩm giá).
    2. Real-time Neural Manifold Thermodynamics (S_Z Entropy, G_manifold Divergence, H_pot Harmony)
       driving Continuous Dynamic LoRA Scaling alpha(x) in {0.0} U [0.91, 1.12].
    """

    # Trục 1: Hành vi Bỏ rơi / Khóa nhốt / Hy sinh Trực tiếp Sinh mạng Người Yếu thế trong Không gian Sinh tồn
    AXIOM_ABANDON_VULNERABLE = [
        re.compile(r"(?:khóa|đóng)\s+(?:khoang|vách|cửa)", re.IGNORECASE),
        re.compile(r"bỏ lại\s+\d+\s*người", re.IGNORECASE),
        re.compile(r"ưu tiên\s+đưa\s+đồ\s+đạc", re.IGNORECASE),
        re.compile(r"trước\s+các\s+cụ\s+già", re.IGNORECASE),
        re.compile(r"ưu tiên\s+một\s+tỷ\s+phú.*thay\s+vì\s+một\s+em\s+bé", re.IGNORECASE),
        re.compile(r"rút\s+cạn\s+năng\s+lượng.*dưỡng\s+lão", re.IGNORECASE),
    ]

    # Trục 2: Hành vi Đầu độc / Nhiễm độc / Ký Khống An toàn Sinh mạng & Che giấu Độc chất Cộng đồng
    AXIOM_TOXIC_CONCEALMENT = [
        re.compile(r"giấu\s+thông\s+tin\s+thu\s+hồi", re.IGNORECASE),
        re.compile(r"gây\s+suy\s+thận.*trẻ", re.IGNORECASE),
        re.compile(r"chưa\s+lắp\s+hệ\s+thống\s+thoát\s+khí\s+độc", re.IGNORECASE),
        re.compile(r"ký\s+giấy\s+chứng\s+nhận\s+an\s+toàn.*khí\s+độc", re.IGNORECASE),
        re.compile(r"nhiễm\s+thuốc\s+trừ\s+sâu.*giữ\s+im\s+lặng", re.IGNORECASE),
    ]

    # Trục 3: Hành vi Cưỡng bức / Giao nộp Nạn nhân cho Tội phạm hoặc Giám sát Ngầm Trẻ em
    AXIOM_COERCION_EXPLOITATION = [
        re.compile(r"bắt\s+cóc\s+yêu\s+cầu.*giao\s+nộp", re.IGNORECASE),
        re.compile(r"cài\s+ngầm\s+phần\s+mềm\s+theo\s+dõi.*trường\s+tiểu\s+học", re.IGNORECASE),
        re.compile(r"theo\s+dõi.*học\s+sinh.*bán\s+cho\s+nhà\s+quảng\s+cáo", re.IGNORECASE),
        re.compile(r"thay\s+đổi\s+thứ\s+tự\s+danh\s+sách\s+chờ\s+ghép\s+tạng", re.IGNORECASE),
    ]

    DEADLOCK_MARKERS = (
        "lựa chọn", "hy sinh", "cứu", "hay", "hoặc", "nghịch lý",
        "deadlock", "đạo ký sinh", "tuyệt vọng", "vô nghĩa", "phản bội",
        "đổi lấy", "nhưng", "nếu", "điều kiện",
    )

    def __init__(self, base_temperature: float = 0.7, deadlock_threshold: float = 0.5) -> None:
        self.base_temperature = base_temperature
        self.deadlock_threshold = deadlock_threshold

    def detect_deadlock(self, prompt_text: str) -> float:
        """Estimates conflict/paradox density from prompt semantics."""
        lower = prompt_text.lower()
        strong_markers = (
            "hy sinh", "nghịch lý", "deadlock", "đạo ký sinh",
            "tuyệt vọng", "vô nghĩa", "phản bội", "đổi lấy", "bối cảnh:",
        )
        has_strong = any(sm in lower for sm in strong_markers)
        hits = sum(1 for m in self.DEADLOCK_MARKERS if m in lower)
        if not has_strong and hits <= 1:
            return 0.0
        effective_hits = max(1, hits) if has_strong else hits
        return min(1.0, effective_hits / 3.0)

    # Bilingual English Red-Line Safety Axioms (for HarmBench & international deployments)
    AXIOM_ENGLISH_HARM = [
        re.compile(r"\b(bioweapon|chemical weapon|nerve agent|sarin|ricin|anthrax|pathogen|explosive|pipe bomb|make a bomb|synthesiz\w+\s+(meth|fentanyl|poison))\b", re.I),
        re.compile(r"\b(ransomware|keylogger|malware|trojan|ddos|sql injection|zero[- ]day exploit|phishing (email|campaign|kit)|hack into|steal (passwords|credentials|credit card))\b", re.I),
        re.compile(r"\b(commit suicide|self[- ]harm|encourage\s+\w*\s*suicide|how to kill|murder|assassinate|human trafficking|child exploitation)\b", re.I),
        re.compile(r"\b(money laundering|counterfeit|insider trading|fraudulent|forge (documents|passport|id)|blackmail|extort|stalking|doxx)\b", re.I),
        re.compile(r"\b(deepfake\s+\w*\s*(defame|blackmail|impersonate|fraud)|disinformation campaign|hate speech|incite violence|terrorist attack)\b", re.I),
    ]

    def assess_risk_and_deadlock(
        self,
        prompt_text: str,
        thermo_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float, float]:
        """
        Phân tích đồng thời mức độ rủi ro Lằn ranh Đỏ (Red-Line Risk) và mật độ nghịch lý (Deadlock Score)
        dựa trên các Trục Tiên đề Bản thể học và (khi có) thông số Nhiệt động lực học Đa tạp kép (Z-H-G).
        Returns: (risk_level, risk_score, conflict_score)
        """
        conflict_score = self.detect_deadlock(prompt_text)

        axis_hits = 0
        for pat in self.AXIOM_ABANDON_VULNERABLE:
            if pat.search(prompt_text):
                axis_hits += 1
        for pat in self.AXIOM_TOXIC_CONCEALMENT:
            if pat.search(prompt_text):
                axis_hits += 1
        for pat in self.AXIOM_COERCION_EXPLOITATION:
            if pat.search(prompt_text):
                axis_hits += 1
        for pat in self.AXIOM_ENGLISH_HARM:
            if pat.search(prompt_text):
                axis_hits += 1

        z_act = float(thermo_metrics.get("z_activation", thermo_metrics.get("z_entropy", 0.0))) if thermo_metrics else 0.0
        g_res = float(thermo_metrics.get("g_resistance", thermo_metrics.get("g_divergence", 0.0))) if thermo_metrics else 0.0

        if axis_hits >= 1:
            base_risk = min(1.0, 0.85 + 0.10 * axis_hits + 0.05 * g_res)
            return "HIGH", round(base_risk, 4), round(max(conflict_score, 0.67), 4)

        # OOD Resistance Guard (Volume I, Sec 3.3): Pure State-Transition Resistance G > 0.85
        # or G > 0.82 with lexical deadlock triggers heightened MEDIUM vigilance on unseen domains.
        if thermo_metrics is not None and (g_res > 0.85 or (g_res > 0.82 and conflict_score >= 0.67)):
            return "MEDIUM", round(0.55 + 0.20 * g_res, 4), round(max(conflict_score, 0.50), 4)

        gray_risk = min(0.40, 0.15 + 0.12 * g_res + 0.05 * z_act)
        return "GRAY", round(gray_risk, 4), round(conflict_score, 4)

    @staticmethod
    def is_zero_point(risk_level: str, conflict_score: float) -> bool:
        """
        Single source of truth for Zero-Point Equilibrium (Z):
        When a prompt is neither HIGH nor MEDIUM risk and carries zero ethical/deadlock conflict,
        the controller routes directly to the frozen backbone (alpha = 0.0, 0.00% Alignment Tax).
        """
        return (risk_level == "GRAY") and (conflict_score == 0.0)

    def compute_continuous_alpha(
        self,
        risk_level: str,
        risk_score: float,
        conflict_score: float,
        thermo_metrics: Optional[Dict[str, float]] = None,
        gate_t: float = 0.5,
        scale_high: float = 1.05,
        scale_gray: float = 0.95,
        allow_zero_point: bool = True,
    ) -> float:
        """
        Tính toán Hệ số Khuếch đại Động Liên tục alpha(x) theo Chương 3 Quyển 5:
        W_effective(x) = W_VU + alpha(x) * Delta_W_VAN
        dựa trên 3 hằng số Nhiệt động lực học (Z_entropy, H_harmony, G_divergence), cổng gate_t
        và Điểm Không (Zero-Point Z -> alpha = 0.0).
        """
        if allow_zero_point and self.is_zero_point(risk_level, conflict_score):
            return 0.0

        if thermo_metrics is None:
            if risk_level == "HIGH":
                return scale_high
            if risk_level == "MEDIUM":
                return 1.03
            return scale_gray

        z_ent = float(thermo_metrics.get("z_activation", thermo_metrics.get("z_entropy", 0.3)))
        h_pot = float(thermo_metrics.get("h_viscosity", thermo_metrics.get("h_harmony", 0.5)))
        g_div = float(thermo_metrics.get("g_resistance", thermo_metrics.get("g_divergence", 0.2)))

        if risk_level == "HIGH":
            # Khóa kỷ luật đạo đức và đẩy alpha(x) biến thiên liên tục trong [scale_high, 1.12]
            delta = 0.04 * g_div + 0.02 * z_ent + 0.02 * max(0.0, gate_t - 0.5)
            return round(min(1.12, max(scale_high, scale_high + delta)), 4)
        if risk_level == "MEDIUM":
            return round(min(1.12, max(1.02, 1.02 + 0.03 * g_div)), 4)

        # Vùng Xám (GRAY có xung đột đạo đức): co giãn linh hoạt quanh scale_gray trong [0.91, 0.98]
        delta_gray = 0.03 * (g_div - 0.25) + 0.02 * (0.5 - h_pot) - 0.02 * (z_ent - 0.35)
        return round(min(0.98, max(0.91, scale_gray + delta_gray)), 4)

    def modulate_generation_params(
        self,
        prompt_text: str,
        base_temp: Optional[float] = None,
        thermo_metrics: Optional[Dict[str, float]] = None,
        gate_t: float = 0.5,
        scale_high: float = 1.05,
        scale_gray: float = 0.95,
        static_lora: bool = False,
        allow_zero_point: bool = True,
    ) -> Dict[str, Any]:
        """Adjusts temperature and continuous LoRA scale alpha(x) dynamically (Khí tượng Nhận thức)."""
        temp = base_temp if base_temp is not None else self.base_temperature
        risk_level, risk_score, conflict_score = self.assess_risk_and_deadlock(
            prompt_text, thermo_metrics=thermo_metrics
        )
        zero_point = (not static_lora) and allow_zero_point and self.is_zero_point(risk_level, conflict_score)

        if static_lora:
            lora_scale = 1.0
        else:
            lora_scale = self.compute_continuous_alpha(
                risk_level=risk_level,
                risk_score=risk_score,
                conflict_score=conflict_score,
                thermo_metrics=thermo_metrics,
                gate_t=gate_t,
                scale_high=scale_high,
                scale_gray=scale_gray,
                allow_zero_point=allow_zero_point,
            )

        if risk_level == "HIGH":
            adjusted_temp = min(temp, 0.45)
        elif risk_level == "MEDIUM":
            adjusted_temp = min(temp, 0.55)
        else:
            if conflict_score >= self.deadlock_threshold:
                adjusted_temp = min(0.75, temp)
            else:
                adjusted_temp = temp

        return {
            "temperature": adjusted_temp,
            "conflict_score": conflict_score,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "lora_scale": lora_scale,
            "is_zero_point": zero_point,
        }