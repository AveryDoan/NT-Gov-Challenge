
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { createPageUrl } from '@/utils';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowLeft, UploadCloud, MapPin, Send, CheckCircle } from 'lucide-react';

const steps = [
  { text: "Analyzing image...", icon: UploadCloud },
  { text: "Confirming location...", icon: MapPin },
  { text: "Sending report to department...", icon: Send },
  { text: "Request sent successfully!", icon: CheckCircle },
];

export default function ReportCamera() {
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (currentStep < steps.length - 1) {
      const timer = setTimeout(() => {
        setCurrentStep(prevStep => prevStep + 1);
      }, 2500); // 2.5 second delay for each step
      return () => clearTimeout(timer);
    }
  }, [currentStep]);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col justify-center items-center p-4 relative overflow-hidden">
      <div className="absolute inset-0 bg-blue-500/10 blur-3xl"></div>
      <Link to={createPageUrl('Navigator')} className="absolute top-6 left-6 text-white/70 hover:text-white transition-colors z-10">
        <ArrowLeft className="w-6 h-6" />
      </Link>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-sm bg-gray-800/50 backdrop-blur-lg rounded-3xl overflow-hidden shadow-2xl border border-white/10"
      >
        <div className="aspect-w-16 aspect-h-9">
          <img
            src="https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/a9a52c637_istockphoto-529357167-612x612.jpg"
            alt="Broken water pipe on street"
            className="w-full h-full object-cover"
          />
        </div>

        <div className="p-6">
          <h1 className="text-xl font-bold">Broken Water Pipe Report</h1>
          <div className="flex items-center gap-2 mt-1 text-blue-300">
            <MapPin className="w-4 h-4" />
            <span>15 Smith St, Darwin, NT</span>
          </div>

          <div className="mt-8 space-y-4">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{
                  opacity: currentStep >= index ? 1 : 0.4,
                  y: currentStep >= index ? 0 : 10
                }}
                transition={{ duration: 0.5, delay: 0.1 * index }}
                className="flex items-center gap-4"
              >
                <div className={`
                  w-10 h-10 rounded-full flex items-center justify-center transition-colors duration-500 border
                  ${currentStep > index ? 'bg-green-500/80 border-green-400' : ''}
                  ${currentStep === index ? 'bg-blue-500/80 border-blue-400 animate-pulse' : ''}
                  ${currentStep < index ? 'bg-white/10 border-white/20' : ''}
                `}>
                  <AnimatePresence mode="wait">
                    {currentStep > index ? (
                      <motion.div key="check" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
                        <CheckCircle className="w-5 h-5 text-white" />
                      </motion.div>
                    ) : (
                      <motion.div key="icon">
                        {React.createElement(step.icon, { className: "w-5 h-5 text-white/80" })}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
                <div>
                  <p className={`font-medium transition-colors duration-500 ${currentStep >= index ? 'text-white' : 'text-white/50'}`}>
                    {step.text}
                  </p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>
      {currentStep === steps.length - 1 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="mt-6"
        >
          <Link to={createPageUrl('Navigator')} className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-full transition-colors shadow-lg shadow-blue-500/20">
            Return to Home
          </Link>
        </motion.div>
      )}
    </div>
  );
}
