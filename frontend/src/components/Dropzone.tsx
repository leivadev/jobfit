import { useId, useState, type ChangeEvent, type DragEvent } from 'react';
import { FILE_VALIDATION_MESSAGES, validateFile } from '../lib/validateFile';

interface DropzoneProps {
  selectedFile: File | null;
  onSelectFile: (file: File) => void;
  onClear: () => void;
}

export function Dropzone({ selectedFile, onSelectFile, onClear }: DropzoneProps) {
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [validationMessage, setValidationMessage] = useState<string | null>(null);
  const inputId = useId();

  function handleFile(file: File) {
    const error = validateFile(file);
    if (error) {
      setValidationMessage(FILE_VALIDATION_MESSAGES[error]);
      return;
    }
    setValidationMessage(null);
    onSelectFile(file);
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) {
      handleFile(file);
    }
    event.target.value = '';
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDraggingOver(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      handleFile(file);
    }
  }

  if (selectedFile) {
    return (
      <div className="flex items-center justify-between gap-3 rounded-md border border-gray-300 px-4 py-3">
        <span className="truncate text-sm text-gray-800">{selectedFile.name}</span>
        <button
          type="button"
          onClick={onClear}
          className="shrink-0 text-sm font-medium text-purple-700 hover:underline"
        >
          Clear
        </button>
      </div>
    );
  }

  return (
    <div>
      <label
        htmlFor={inputId}
        onDragOver={(event) => {
          event.preventDefault();
          setIsDraggingOver(true);
        }}
        onDragLeave={() => setIsDraggingOver(false)}
        onDrop={handleDrop}
        className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed px-4 py-10 text-center text-sm text-gray-600 ${
          isDraggingOver ? 'border-purple-500 bg-purple-50' : 'border-gray-300'
        }`}
      >
        <span>Drag and drop your CV here, or click to browse</span>
        <span className="text-xs text-gray-400">PDF, DOCX, or TXT, up to 5 MB</span>
      </label>
      <input
        id={inputId}
        type="file"
        accept=".pdf,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/plain"
        onChange={handleInputChange}
        className="sr-only"
      />
      {validationMessage && <p className="mt-2 text-sm text-red-600">{validationMessage}</p>}
    </div>
  );
}
