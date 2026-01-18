import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, Check } from 'lucide-react'

export default function FileUpload({ onFileUpload, loading, fileInfo }) {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles[0])
    }
  }, [onFileUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/wav': ['.wav'],
      'audio/mpeg': ['.mp3'],
      'text/csv': ['.csv'],
    },
    multiple: false,
    disabled: loading,
  })

  return (
    <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-8">
      <h2 className="text-xs font-semibold mb-4 text-biotuner-light/60 uppercase tracking-wider flex items-center gap-2">
        <Upload className="w-4 h-4" />
        Upload Data Source
      </h2>
      
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-12 text-center transition-all duration-300 cursor-pointer overflow-hidden
          ${isDragActive
            ? 'border-biotuner-primary bg-biotuner-primary/5'
            : 'border-biotuner-dark-600 hover:border-biotuner-primary/50 bg-biotuner-dark-800/50'
          }
          ${loading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        <div className="relative z-10">
          <div className="flex justify-center mb-4">
            <div className="w-16 h-16 rounded-full bg-gradient-to-br from-biotuner-primary to-biotuner-secondary flex items-center justify-center">
              <Upload className="w-8 h-8 text-biotuner-dark-900" />
            </div>
          </div>
          
          {loading ? (
            <div>
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-biotuner-primary border-t-transparent mx-auto mb-4"></div>
              <p className="text-biotuner-light">Processing file...</p>
            </div>
          ) : isDragActive ? (
            <p className="text-biotuner-primary text-lg font-medium">Drop file here...</p>
          ) : (
            <div>
              <p className="text-biotuner-light mb-2 text-lg">
                Drag & drop your file here, or <span className="text-biotuner-primary">click to browse</span>
              </p>
              <p className="text-biotuner-light/40 text-sm">
                Supported formats: WAV, MP3, CSV (max 50MB)
              </p>
            </div>
          )}
        </div>
      </div>

      {fileInfo && (
        <div className="mt-4 p-4 bg-biotuner-dark-800 rounded-lg border border-biotuner-dark-600">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-biotuner-accent to-biotuner-primary flex items-center justify-center">
              <Check className="w-5 h-5 text-biotuner-dark-900" />
            </div>
            <div className="flex-1">
              <p className="text-biotuner-light font-medium truncate">{fileInfo.filename}</p>
              <p className="text-biotuner-light/40 text-sm">
                {fileInfo.duration?.toFixed(2)}s • {fileInfo.sampling_rate} Hz
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
