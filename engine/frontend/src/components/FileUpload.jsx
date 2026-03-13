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
    <div className="bg-biotuner-dark-900 rounded-lg border border-biotuner-dark-600 p-4 sm:p-6 lg:p-8">
      <h2 className="text-xs font-semibold mb-4 text-biotuner-light/60 uppercase tracking-wider flex items-center gap-2">
        <Upload className="w-4 h-4" />
        Upload Data Source
      </h2>
      
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-6 sm:p-8 lg:p-12 text-center transition-all duration-300 cursor-pointer overflow-hidden
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
            <div className="w-12 h-12 sm:w-16 sm:h-16 rounded-full bg-gradient-to-br from-biotuner-primary to-biotuner-secondary flex items-center justify-center">
              <Upload className="w-6 h-6 sm:w-8 sm:h-8 text-biotuner-dark-900" />
            </div>
          </div>
          
          {loading ? (
            <div>
              <div className="animate-spin rounded-full h-10 w-10 sm:h-12 sm:w-12 border-4 border-biotuner-primary border-t-transparent mx-auto mb-4"></div>
              <p className="text-biotuner-light text-sm sm:text-base">Processing file...</p>
            </div>
          ) : isDragActive ? (
            <p className="text-biotuner-primary text-base sm:text-lg font-medium">Drop file here...</p>
          ) : (
            <div>
              <p className="text-biotuner-light mb-2 text-base sm:text-lg">
                <span className="hidden sm:inline">Drag & drop your file here, or </span>
                <span className="text-biotuner-primary">
                  <span className="sm:hidden">Tap to upload</span>
                  <span className="hidden sm:inline">click to browse</span>
                </span>
              </p>
              <p className="text-biotuner-light/40 text-xs sm:text-sm">
                Supported: WAV, MP3, CSV (max 50MB)
              </p>
            </div>
          )}
        </div>
      </div>

      {fileInfo && (
        <div className="mt-4 p-3 sm:p-4 bg-biotuner-dark-800 rounded-lg border border-biotuner-dark-600">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg bg-gradient-to-br from-biotuner-accent to-biotuner-primary flex items-center justify-center flex-shrink-0">
              <Check className="w-4 h-4 sm:w-5 sm:h-5 text-biotuner-dark-900" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-biotuner-light font-medium truncate text-sm sm:text-base">{fileInfo.filename}</p>
              <p className="text-biotuner-light/40 text-xs sm:text-sm">
                {fileInfo.duration?.toFixed(2)}s • {fileInfo.sampling_rate} Hz
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
