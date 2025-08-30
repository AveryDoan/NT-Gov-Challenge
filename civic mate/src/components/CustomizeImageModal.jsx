import React, { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Upload, Camera, X } from 'lucide-react';
import { UploadFile } from '@/api/integrations';
import { UserLifeEventImage } from '@/api/entities';

export default function CustomizeImageModal({ show, onClose, eventKey, eventTitle, currentImage, onImageUpdated }) {
  const [uploading, setUploading] = useState(false);
  const [selectedImage, setSelectedImage] = useState(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedImage(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedImage) return;

    setUploading(true);
    try {
      const { file_url } = await UploadFile({ file: selectedImage });
      
      // Check if user already has a custom image for this event
      const existingImages = await UserLifeEventImage.filter({
        event_key: eventKey,
        created_by: (await UserLifeEventImage.User?.me())?.email
      });

      if (existingImages.length > 0) {
        // Update existing
        await UserLifeEventImage.update(existingImages[0].id, {
          custom_image_url: file_url
        });
      } else {
        // Create new
        await UserLifeEventImage.create({
          event_key: eventKey,
          custom_image_url: file_url
        });
      }
      
      onImageUpdated(eventKey, file_url);
      onClose();
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
      setSelectedImage(null);
    }
  };

  const handleRemoveCustomImage = async () => {
    try {
      const existingImages = await UserLifeEventImage.filter({
        event_key: eventKey,
        created_by: (await UserLifeEventImage.User?.me())?.email
      });

      if (existingImages.length > 0) {
        await UserLifeEventImage.delete(existingImages[0].id);
        onImageUpdated(eventKey, null);
        onClose();
      }
    } catch (error) {
      console.error('Remove failed:', error);
    }
  };

  return (
    <Dialog open={show} onOpenChange={onClose}>
      <DialogContent className="bg-white border-gray-200 text-gray-900 max-w-md mx-auto">
        <DialogHeader>
          <DialogTitle className="text-xl mb-4 text-center">
            Customize "{eventTitle}"
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-6">
          {/* Current Image Preview */}
          <div className="text-center">
            <div className="w-40 h-40 mx-auto rounded-2xl overflow-hidden border-2 border-gray-200 mb-4">
              <img 
                src={selectedImage ? URL.createObjectURL(selectedImage) : currentImage} 
                alt="Event preview"
                className="w-full h-full object-cover"
              />
            </div>
            
            {selectedImage && (
              <p className="text-sm text-blue-600 mb-4">Preview of new image</p>
            )}
          </div>

          {/* Upload Section */}
          <div className="space-y-4">
            <label className="block">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
              />
              <div className="w-full p-4 border-2 border-dashed border-gray-300 rounded-xl text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-colors">
                <Camera className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-sm text-gray-600">
                  {selectedImage ? selectedImage.name : 'Click to select image'}
                </p>
              </div>
            </label>

            <div className="flex gap-3">
              {selectedImage && (
                <Button 
                  onClick={handleUpload}
                  disabled={uploading}
                  className="flex-1 bg-blue-500 hover:bg-blue-600 text-white"
                >
                  {uploading ? (
                    <>
                      <Upload className="w-4 h-4 mr-2 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Upload Image
                    </>
                  )}
                </Button>
              )}
              
              {currentImage && (
                <Button 
                  onClick={handleRemoveCustomImage}
                  variant="outline"
                  className="flex-1"
                >
                  <X className="w-4 h-4 mr-2" />
                  Use Default
                </Button>
              )}
            </div>
          </div>

          <Button 
            onClick={onClose}
            variant="outline"
            className="w-full"
          >
            Cancel
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}