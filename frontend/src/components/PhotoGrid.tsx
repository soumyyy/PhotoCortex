import Image from 'next/image';
import { useState, useEffect } from 'react';
import { ProcessingResult } from '@/types';
import { InfoIcon } from '@/components/icons/InfoIcon';

interface PhotoGridProps {
  processedImages: ProcessingResult[];
}

export function PhotoGrid({ processedImages }: PhotoGridProps) {
  const [hoveredImage, setHoveredImage] = useState<string | null>(null);
  const [showMetadata, setShowMetadata] = useState<string | null>(null);

  useEffect(() => {
    console.log('Processed Images:', processedImages);
  }, [processedImages]);

  const formatDate = (isoString: string) => {
    return new Date(isoString).toLocaleString();
  };

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
      {processedImages.map((result, idx) => (
        <div 
          key={idx} 
          className="relative group"
          onMouseEnter={() => setHoveredImage(result.filename)}
          onMouseLeave={() => {
            setHoveredImage(null);
            setShowMetadata(null);
          }}
        >
          {result.full_image && (
            <>
              <div className="aspect-square relative">
                <Image
                  src={result.full_image}
                  alt={result.filename}
                  className="rounded-lg object-cover"
                  fill
                />
                {hoveredImage === result.filename && (
                  <button
                    className="absolute top-2 right-2 p-2 bg-black/50 rounded-full hover:bg-black/70 transition-colors"
                    onClick={(e) => {
                      e.stopPropagation();
                      setShowMetadata(showMetadata === result.filename ? null : result.filename);
                    }}
                  >
                    <InfoIcon className="w-5 h-5 text-white" />
                  </button>
                )}
              </div>
              {showMetadata === result.filename && (
                <div className="absolute top-0 left-0 right-0 bottom-0 bg-black/90 p-4 rounded-lg overflow-y-auto">
                  <div className="text-white space-y-3">
                    <div>
                      <h3 className="font-semibold text-lg mb-1">Scene Information</h3>
                      <p className="text-sm">{result.info}</p>
                    </div>

                    <div>
                      <h3 className="font-semibold text-lg mb-1">Basic Information</h3>
                      <div className="text-sm">
                        <p>Filename: {result.metadata.basic.filename}</p>
                        <p>Size: {result.metadata.basic.file_size}</p>
                        <p>Dimensions: {result.metadata.basic.dimensions}</p>
                        <p>Format: {result.metadata.basic.format}</p>
                        <p>Created: {formatDate(result.metadata.basic.created)}</p>
                        <p>Modified: {formatDate(result.metadata.basic.modified)}</p>
                      </div>
                    </div>
                    {Object.keys(result.metadata.exif).length > 0 && (
                      <div>
                        <h3 className="font-semibold text-lg mb-1">Camera Info</h3>
                        <div className="text-sm">
                          {result.metadata.exif.make && <p>Make: {result.metadata.exif.make}</p>}
                          {result.metadata.exif.model && <p>Model: {result.metadata.exif.model}</p>}
                          {result.metadata.exif.exposuretime && <p>Exposure: {result.metadata.exif.exposuretime}</p>}
                          {result.metadata.exif.fnumber && <p>F-Stop: {result.metadata.exif.fnumber}</p>}
                          {result.metadata.exif.iso && <p>ISO: {result.metadata.exif.iso}</p>}
                        </div>
                      </div>
                    )}
                    {Object.keys(result.metadata.gps).length > 0 && (
                      <div>
                        <h3 className="font-semibold text-lg mb-1">Location</h3>
                        <div className="text-sm">
                          <p>Lat: {result.metadata.gps.latitude}</p>
                          <p>Long: {result.metadata.gps.longitude}</p>
                          {result.metadata.gps.altitude && (
                            <p>Altitude: {result.metadata.gps.altitude}</p>
                          )}
                        </div>
                      </div>
                    )}

                    {result.faces.length > 0 && (
                      <div>
                        <h3 className="font-semibold text-lg mb-1">Faces</h3>
                        <p className="text-sm">{result.faces.length} faces detected</p>
                      </div>
                    )}
                  </div>
                </div>
              )}
              {hoveredImage === result.filename && !showMetadata && (
                <div className="absolute bottom-0 left-0 right-0 bg-black/70 p-2 rounded-b-lg">
                  <p className="text-sm text-white">{result.filename}</p>
                  <p className="text-xs text-gray-300">
                    {result.faces.length} faces detected
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      ))}
    </div>
  );
} 