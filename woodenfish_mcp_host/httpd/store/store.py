from abc import ABC, abstractmethod

from fastapi import UploadFile

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".rtf",
    ".odt",
    ".html",
    ".csv",
    ".epub",
}


class Store(ABC):
    """存储操作的抽象基类。"""

    @abstractmethod
    async def upload_files(
        self,
        files: list[UploadFile],
        file_paths: list[str],
    ) -> tuple[list[str], list[str]]:
        """上传文件到存储。"""

    @abstractmethod
    async def get_image(self, file_path: str) -> str:
        """从存储获取 base64 编码的图片。"""

    @abstractmethod
    async def get_document(self, file_path: str) -> tuple[str, str | None]:
        """从存储获取 base64 编码的文档。

        Args:
            file_path: The path to the document.

        Returns:
            tuple[str, str | None]: The base64 encoded document and the mime type.
        """
