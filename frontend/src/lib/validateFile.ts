export const MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024;

export const ACCEPTED_MIME_TYPES = [
  'application/pdf',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'text/plain',
];

export type FileValidationError = 'too-large' | 'unsupported-type';

export function validateFile(file: File): FileValidationError | null {
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return 'too-large';
  }
  if (!ACCEPTED_MIME_TYPES.includes(file.type)) {
    return 'unsupported-type';
  }
  return null;
}

export const FILE_VALIDATION_MESSAGES: Record<FileValidationError, string> = {
  'too-large': 'That file is over 5 MB. Choose a smaller file.',
  'unsupported-type': 'Unsupported file type. Use a .pdf, .docx, or .txt file.',
};
