from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

opts = PdfPipelineOptions()
opts.do_table_structure = True
opts.table_structure_options.mode = TableFormerMode.ACCURATE
opts.table_structure_options.do_cell_matching = True
opts.accelerator_options = AcceleratorOptions(num_threads=1, device=AcceleratorDevice.CPU)

DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=opts,
            backend=DoclingParseDocumentBackend
        )
    }
)
print("Modelos descargados correctamente")
