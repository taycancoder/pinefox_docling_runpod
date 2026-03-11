import runpod
from pathlib import Path
import base64
import tempfile
import requests
import gc
import threading
import torch

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, TableFormerMode, AcceleratorDevice, AcceleratorOptions
)
from docling.datamodel.base_models import InputFormat
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# ─────────────────────────────────────────────────────────────────────
# DEFAULTS GLOBALES — se usan si el backend no envía el parámetro
# ─────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "num_threads":          8,          # Hilos CPU para preprocesamiento
    "do_ocr":               False,      # OCR desactivado (PDFs sin texto ya filtrados)
    "table_mode":           "accurate", # "accurate" | "fast"
    "do_table_structure":   True,       # Extracción de tablas activa
    "do_cell_matching":     True,       # Fusión de celdas (tablas financieras)
    "generate_page_images": False,      # No generar imágenes de página
    "generate_pic_images":  False,      # No generar imágenes de figuras
    "timeout_cuda_accurate":800,        # Segundos máx para backend cuda_accurate
    "timeout_cuda_fast":    400,        # Segundos máx para backend cuda_fast
    "timeout_cpu_fallback": 900,        # Segundos máx para backend cpu_fallback
    "max_file_size_mb":     500,        # Tamaño máximo permitido del PDF
    "min_markdown_chars":   100,        # Mínimo de chars para considerar conversión válida
    "force_backend":        None,       # None | "cuda_accurate" | "cuda_fast" | "cpu_fallback"
}

TABLE_MODE_MAP = {
    "accurate": TableFormerMode.ACCURATE,
    "fast":     TableFormerMode.FAST,
}

# ─────────────────────────────────────────────────────────────────────
# CONSTRUCCIÓN DINÁMICA DE PIPELINE
# ─────────────────────────────────────────────────────────────────────
def _build_pipeline(cfg, device, force_ocr=None):
    opts = PdfPipelineOptions()

    opts.do_ocr                              = force_ocr if force_ocr is not None else cfg["do_ocr"]
    opts.generate_page_images               = cfg["generate_page_images"]
    opts.generate_picture_images            = cfg["generate_pic_images"]
    opts.do_table_structure                 = cfg["do_table_structure"]
    opts.table_structure_options.mode       = TABLE_MODE_MAP.get(cfg["table_mode"], TableFormerMode.ACCURATE)
    opts.table_structure_options.do_cell_matching = cfg["do_cell_matching"]
    opts.accelerator_options                = AcceleratorOptions(
        num_threads=cfg["num_threads"],
        device=device
    )
    return opts


def _build_converters(cfg):
    """Construye los 3 convertidores usando la config del job actual."""
    cuda_accurate = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=_build_pipeline(cfg, AcceleratorDevice.CUDA),
            backend=DoclingParseDocumentBackend
        )}
    )
    cuda_fast = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=_build_pipeline(cfg, AcceleratorDevice.CUDA, force_ocr=False),
            backend=DoclingParseDocumentBackend
        )}
    )
    # CPU fallback: hereda do_ocr del cfg (si el job envía do_ocr=true, aquí también aplica)
    cpu_fallback = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=_build_pipeline(cfg, AcceleratorDevice.CPU),
            backend=PyPdfiumDocumentBackend
        )}
    )
    return [
        {"name": "cuda_accurate", "converter": cuda_accurate, "timeout": cfg["timeout_cuda_accurate"]},
        {"name": "cuda_fast",     "converter": cuda_fast,     "timeout": cfg["timeout_cuda_fast"]},
        {"name": "cpu_fallback",  "converter": cpu_fallback,  "timeout": cfg["timeout_cpu_fallback"]},
    ]


# ─────────────────────────────────────────────────────────────────────
# EJECUCIÓN CON TIMEOUT REAL (threading)
# ─────────────────────────────────────────────────────────────────────
def _run_conversion(converter, path_str, timeout_sec):
    result_holder = [None, None]  # [resultado, excepcion]

    def _worker():
        try:
            res = converter.convert(path_str, raises_on_error=True)
            result_holder[0] = res.document.export_to_markdown()
        except Exception as e:
            result_holder[1] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        raise TimeoutError(f"Conversión excedió {timeout_sec}s")
    if result_holder[1] is not None:
        raise result_holder[1]

    return result_holder[0]


def _clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def categorize_error(error_obj):
    err_str = str(error_obj).lower()
    if "cuda out of memory" in err_str: return "CUDA_OOM"
    if "timeout"           in err_str: return "TIMEOUT"
    if "pdfium"            in err_str: return "PDF_CORRUPT"
    if "ocr"               in err_str: return "OCR_ERROR"
    return "CONVERSION_ERROR"


# ─────────────────────────────────────────────────────────────────────
# HANDLER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────
def handler(job):
    job_input  = job.get('input', {})
    pdf_url    = job_input.get('pdf_url')
    pdf_base64 = job_input.get('pdf_base64')
    file_name  = job_input.get('fileName', 'archivo_desconocido')

    if not pdf_url and not pdf_base64:
        return {
            "error": "Falta pdf_url o pdf_base64",
            "fileName": file_name,
            "error_type": "VALIDATION_ERROR"
        }

    # ── Construir config mezclando DEFAULTS + parámetros del job ──
    def get(key, cast=None):
        val = job_input.get(key, DEFAULTS[key])
        if cast and val is not None:
            try:
                return cast(val)
            except:
                return DEFAULTS[key]
        return val

    cfg = {
        "num_threads":           get("num_threads",           int),
        "do_ocr":                get("do_ocr",                bool),
        "table_mode":            get("table_mode"),
        "do_table_structure":    get("do_table_structure",    bool),
        "do_cell_matching":      get("do_cell_matching",      bool),
        "generate_page_images":  get("generate_page_images",  bool),
        "generate_pic_images":   get("generate_pic_images",   bool),
        "timeout_cuda_accurate": get("timeout_cuda_accurate", int),
        "timeout_cuda_fast":     get("timeout_cuda_fast",     int),
        "timeout_cpu_fallback":  get("timeout_cpu_fallback",  int),
        "max_file_size_mb":      get("max_file_size_mb",      int),
        "min_markdown_chars":    get("min_markdown_chars",    int),
        "force_backend":         get("force_backend"),
    }

    # Validar table_mode
    if cfg["table_mode"] not in TABLE_MODE_MAP:
        cfg["table_mode"] = DEFAULTS["table_mode"]

    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "document.pdf"

        # ── Descarga ──
        try:
            if pdf_url:
                response = requests.get(pdf_url, stream=True, timeout=(15, 300))
                response.raise_for_status()
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                with open(pdf_path, "wb") as f:
                    f.write(base64.b64decode(pdf_base64))

        except Exception as e:
            return {"error": f"Descarga fallida: {e}", "error_type": "DOWNLOAD_ERROR", "fileName": file_name}

        # ── Validar tamaño ──
        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if file_size_mb > cfg["max_file_size_mb"]:
            return {
                "error": f"PDF demasiado grande: {file_size_mb:.1f}MB (max: {cfg['max_file_size_mb']}MB)",
                "error_type": "SIZE_ERROR",
                "fileName": file_name
            }

        # ── Construir convertidores con config del job ──
        backends = _build_converters(cfg)

        # ── Filtrar por force_backend si se especificó ──
        if cfg["force_backend"]:
            backends = [b for b in backends if b["name"] == cfg["force_backend"]]
            if not backends:
                return {
                    "error": f"force_backend '{cfg['force_backend']}' no es válido",
                    "error_type": "VALIDATION_ERROR",
                    "fileName": file_name
                }

        # ── Cascade de conversión ──
        errors_log = []

        for backend in backends:
            try:
                markdown_text = _run_conversion(
                    backend["converter"],
                    str(pdf_path),
                    backend["timeout"]
                )

                if not markdown_text or len(markdown_text.strip()) < cfg["min_markdown_chars"]:
                    raise ValueError(
                        f"Markdown insuficiente ({len(markdown_text or '')} chars) — "
                        f"mínimo requerido: {cfg['min_markdown_chars']}"
                    )

                return {
                    "status":          "success",
                    "fileName":        file_name,
                    "backend_used":    backend["name"],
                    "file_size_mb":    round(file_size_mb, 2),
                    "markdown_length": len(markdown_text),
                    "config_used":     cfg,
                    "markdown":        markdown_text
                }

            except Exception as e:
                errors_log.append({
                    "backend":    backend["name"],
                    "error":      str(e),
                    "error_type": categorize_error(e)
                })
                _clear_vram()
                continue

        return {
            "status":     "error",
            "error":      "Fallo en todos los backends",
            "error_type": "ALL_BACKENDS_FAILED",
            "fileName":   file_name,
            "errors_log": errors_log
        }


runpod.serverless.start({"handler": handler})