
import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from "@/components/ui/dropdown-menu";
import { Home, Wallet, Mail, Grid3X3, Compass, Baby, Briefcase, Heart, MapPin, Check, Square, Mic, MicOff, ChevronRight, Settings, Landmark, Users, HeartPulse, FileText, BadgeInfo, CircleDollarSign, TrendingUp, X, Camera, Wand2, User, Siren, HelpCircle, LogOut, FileQuestion, Loader2 } from 'lucide-react';
import { UserLifeEventImage } from '@/api/entities';
import CustomizeImageModal from '../components/CustomizeImageModal';
import { motion, AnimatePresence } from 'framer-motion';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';
import { createPageUrl } from '@/utils';
import { Link } from 'react-router-dom';
import { InvokeLLM } from '@/api/integrations';

const LIFE_EVENTS = {
  baby: {
    title: 'Had a baby',
    icon: Baby,
    illustration: 'https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?w=200&h=200&fit=crop&crop=face',
    description: 'Register birth and access family benefits',
    keywords: ['baby', 'birth', 'newborn', 'pregnant', 'child', 'born', 'had a baby', 'expecting'],
    headerIcon: '🎉',
    headerText: 'Congratulations on your new family member!',
    tasks: [
      'Register birth',
      'Link Medicare',
      'Check Parental Leave Pay',
      'Add child to Family Tax Benefit',
      'Apply for Child Care Subsidy',
      'Update address',
      'Update tax details'
    ],
    taskPriorities: ['high', 'high', 'medium', 'low', 'medium', 'low', 'low']
  },
  job: {
    title: 'Lost a job',
    icon: Briefcase,
    illustration: 'https://images.unsplash.com/photo-1521737604893-d14cc237f11d?w=200&h=200&fit=crop&crop=center',
    description: 'Access unemployment benefits and job services',
    keywords: ['job', 'work', 'unemployed', 'fired', 'redundant', 'employment', 'lost my job', 'lost a job', 'retrenched'],
    headerIcon: '💼',
    headerText: 'Support for finding your next opportunity.',
    tasks: [
      'Register JobSeeker',
      'Update income with ATO',
      'Check healthcare concessions',
      'Update address',
      'Access employment services',
      'Review superannuation',
      'Update Medicare'
    ],
    taskPriorities: ['high', 'high', 'medium', 'medium', 'low', 'low', 'low']
  },
  carer: {
    title: 'Became a carer',
    icon: Heart,
    illustration: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=200&h=200&fit=crop&crop=center',
    description: 'Access carer support and benefits',
    keywords: ['carer', 'caring', 'disability', 'elderly', 'support', 'assistance', 'became a carer', 'look after'],
    headerIcon: '❤️',
    headerText: "We're here to support you as a carer.",
    tasks: [
      'Apply Carer Payment',
      'Register Carer Allowance',
      'Get Recognition Statement',
      'Update Medicare',
      'Check disability support',
      'Access respite care',
      'Update tax details'
    ],
    taskPriorities: ['high', 'high', 'medium', 'medium', 'low', 'low', 'low']
  },
  moved: {
    title: 'Moved address',
    icon: MapPin,
    illustration: 'https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=200&h=200&fit=crop&crop=center',
    description: 'Update your address with government services',
    keywords: ['moved', 'move', 'address', 'relocate', 'new home', 'moved address', 'changed address', 'relocating'],
    headerIcon: '📦',
    headerText: "Let's get your new address updated.",
    tasks: [
      'Update Electoral Commission',
      'Change Centrelink address',
      'Notify ATO',
      'Update bank details',
      'Change driver licence',
      'Update Australia Post'
    ],
    taskPriorities: ['high', 'high', 'high', 'medium', 'medium', 'low']
  }
};

const TASK_ICONS = {
  'Register birth': '📄',
  'Link Medicare': '🏥',
  'Check Parental Leave Pay': '💰',
  'Add child to Family Tax Benefit': '👨‍👩‍👧‍👦',
  'Apply for Child Care Subsidy': '🏫',
  'Update address': '📍',
  'Update tax details': '📊',
  'Register JobSeeker': '💼',
  'Update income with ATO': '💰',
  'Check healthcare concessions': '🏥',
  'Access employment services': '👔',
  'Review superannuation': '🏦',
  'Update Medicare': '🏥',
  'Apply Carer Payment': '💳',
  'Register Carer Allowance': '❤️',
  'Get Recognition Statement': '📜',
  'Check disability support': '🤝',
  'Access respite care': '🏠',
  'Update Electoral Commission': '🗳️',
  'Change Centrelink address': '🏢',
  'Notify ATO': '📋',
  'Update bank details': '🏦',
  'Change driver licence': '🚗',
  'Update Australia Post': '📮'
};

const servicesData = [
  { name: 'Australian Taxation Office', icon: Landmark, status: 'Connected' },
  { name: 'Centrelink', icon: Users, status: 'Connected' },
  { name: 'Medicare', icon: HeartPulse, status: 'Connected' },
  { name: 'My Health Record', icon: FileText, status: 'Connected' },
  { name: 'Workforce Australia', icon: Briefcase, status: 'Connected' },
  { name: 'Individual Healthcare Identifiers', icon: BadgeInfo, status: 'Connected' }
];

export default function Navigator() {
  const [activeTab, setActiveTab] = useState('home');
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [showVoiceModal, setShowVoiceModal] = useState(false);
  const [voiceTranscript, setVoiceTranscript] = useState('');
  const [customImages, setCustomImages] = useState({});
  const [showCustomizeModal, setShowCustomizeModal] = useState(false);
  const [customizeEventKey, setCustomizeEventKey] = useState(null);
  const [completedTasks, setCompletedTasks] = useState(new Set());
  const [modalCompletedTasks, setModalCompletedTasks] = useState(new Set()); // New state for modal tasks
  const [chatMessage, setChatMessage] = useState('');
  const [activeEventContext, setActiveEventContext] = useState(null);
  const [isFabMenuOpen, setIsFabMenuOpen] = useState(false);
  // Removed showCameraModal, as it's now a navigation link
  const [showIdCardModal, setShowIdCardModal] = useState(false);
  const [showEmergencyModal, setShowEmergencyModal] = useState(false);
  const [showTaskModal, setShowTaskModal] = useState(false);
  const [selectedTaskIndex, setSelectedTaskIndex] = useState(null);
  const [taskFormData, setTaskFormData] = useState({ name: '', email: '', phone: '' });
  const [isAgentThinking, setIsAgentThinking] = useState(false);
  const [suggestionsVisible, setSuggestionsVisible] = useState(false);

  // Quick suggestion options
  const quickSuggestions = [
    { text: "I just had a baby", icon: "🍼" },
    { text: "I lost my job", icon: "💼" },
    { text: "I'm caring for someone", icon: "❤️" },
    { text: "I moved to a new house", icon: "🏠" }
  ];

  useEffect(() => {
    loadCustomImages();
  }, []);

  const loadCustomImages = async () => {
    try {
      const images = await UserLifeEventImage.list();
      const imageMap = {};
      images.forEach((img) => {
        imageMap[img.event_key] = img.custom_image_url;
      });
      setCustomImages(imageMap);
    } catch (error) {
      console.error('Failed to load custom images:', error);
    }
  };

  const handleImageUpdated = (eventKey, imageUrl) => {
    setCustomImages((prev) => ({
      ...prev,
      [eventKey]: imageUrl
    }));
  };

  const handleVoiceComplete = (detectedEvent, transcript) => {
    setVoiceTranscript(transcript);
    setShowVoiceModal(false);
    setSelectedEvent(detectedEvent);
  };

  const handleTaskClick = (taskIndex) => {
    setSelectedTaskIndex(taskIndex);
    setShowTaskModal(true);
  };

  const handleTaskSubmit = async (e) => {
    e.preventDefault();
    // Simulate submission
    setShowTaskModal(false);

    // Mark task as completed
    toggleTaskComplete(selectedTaskIndex);

    // Reset form
    setTaskFormData({ name: '', email: '', phone: '' });
    setSelectedTaskIndex(null);
  };

  const toggleTaskComplete = (taskIndex) => {
    setCompletedTasks((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(taskIndex)) {
        newSet.delete(taskIndex);
      } else {
        newSet.add(taskIndex);
      }
      return newSet;
    });
  };

  const toggleModalTaskComplete = (taskIndex) => {
    setModalCompletedTasks((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(taskIndex)) {
        newSet.delete(taskIndex);
      } else {
        newSet.add(taskIndex);
      }
      return newSet;
    });
  };

  const handleSuggestionClick = (suggestionText) => {
    setSuggestionsVisible(false); // Hide on click
    setChatMessage(suggestionText);
    // Automatically submit the suggestion
    handleChatSubmit({ preventDefault: () => {} }, suggestionText);
  };

  const handleChatSubmit = async (e, directMessage = null) => {
    e.preventDefault();
    const message = directMessage || chatMessage.trim();
    if (!message) return;

    setIsAgentThinking(true);
    setChatMessage('');

    try {
      const prompt = `A user has described a life event: "${message}". 

Based on this message, identify which life event category it matches:
- 'baby': having a baby, pregnancy, birth, newborn
- 'job': losing job, unemployment, fired, redundant, jobless
- 'carer': becoming a carer, caring for elderly/disabled person
- 'moved': moving house, change of address, relocating

Examples:
- "I just had a baby" -> baby
- "I lost my job" -> job
- "I'm caring for my elderly mother" -> carer  
- "I moved to a new house" -> moved

Return only one word: baby, job, carer, moved, or none`;

      console.log("Sending prompt to AI:", prompt);

      const response = await InvokeLLM({
        prompt,
        response_json_schema: {
          type: "object",
          properties: {
            event: {
              type: "string",
              enum: ["baby", "job", "carer", "moved", "none"]
            }
          },
          required: ["event"]
        }
      });

      console.log("AI Response:", response);

      const detectedEventKey = response?.event;

      if (detectedEventKey && detectedEventKey !== 'none' && LIFE_EVENTS[detectedEventKey]) {
        console.log("Setting active event context:", detectedEventKey);
        setActiveEventContext(detectedEventKey);
        setCompletedTasks(new Set()); // Reset tasks for the new context
      } else {
        console.log("No event matched or invalid response:", detectedEventKey);
        // Fallback: try to match keywords manually
        const lowerMessage = message.toLowerCase();
        let matchedEvent = null;
        
        for (const [eventKey, eventData] of Object.entries(LIFE_EVENTS)) {
          if (eventData.keywords.some(keyword => lowerMessage.includes(keyword))) {
            matchedEvent = eventKey;
            break;
          }
        }
        
        if (matchedEvent) {
          console.log("Fallback keyword match:", matchedEvent);
          setActiveEventContext(matchedEvent);
          setCompletedTasks(new Set());
        } else {
          console.log("No fallback match found");
          setActiveEventContext(null);
        }
      }
    } catch (error) {
      console.error("Error invoking AI agent:", error);
      // Fallback: try to match keywords manually
      const lowerMessage = message.toLowerCase();
      let matchedEvent = null;
      
      for (const [eventKey, eventData] of Object.entries(LIFE_EVENTS)) {
        if (eventData.keywords.some(keyword => lowerMessage.includes(keyword))) {
          matchedEvent = eventKey;
          break;
        }
      }
      
      if (matchedEvent) {
        console.log("Fallback keyword match:", matchedEvent);
        setActiveEventContext(matchedEvent);
        setCompletedTasks(new Set());
      } else {
        console.log("No fallback match found");
        setActiveEventContext(null);
      }
    } finally {
      setIsAgentThinking(false);
    }
  };

  const QuickActionCard = ({ eventKey, event }) => {
    const imageUrl = customImages[eventKey] || event.illustration;

    return (
      <div className="relative">
        <button
          onClick={() => {
            setSelectedEvent(eventKey); // For modal
            setActiveEventContext(eventKey); // For main page content
            setCompletedTasks(new Set()); // Reset tasks
          }}
          className="relative w-full aspect-square rounded-2xl overflow-hidden hover:shadow-lg transition-all duration-200 border-2 border-white hover:border-blue-400"
          style={{
            backgroundImage: `url(${imageUrl})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center'
          }}>

          {/* Overlay text removed for a cleaner look */}
        </button>
      </div>);

  };

  const inboxItems = [
    {
      id: 1,
      title: "Remind to renew the MVR",
      description: "Your Motor Vehicle Registration is due for renewal",
      icon: "🚗",
      time: "2 hours ago",
      unread: true
    },
    {
      id: 2,
      title: "Remind to submit tax return",
      description: "Don't forget to submit your annual tax return",
      icon: "📊",
      time: "1 day ago",
      unread: true
    },
    {
      id: 3,
      title: "You are eligible for medicare",
      description: "Complete your Medicare enrollment to access healthcare benefits",
      icon: "🏥",
      time: "3 days ago",
      unread: false
    },
    {
      id: 4,
      title: "You should study the TAFE course to improve your skill",
      description: "Explore TAFE courses that match your career goals",
      icon: "🎓",
      time: "1 week ago",
      unread: false
    }];


  const portfolioData = [
    { name: 'Jan', value: 45000 },
    { name: 'Feb', value: 46500 },
    { name: 'Mar', value: 48000 },
    { name: 'Apr', value: 47500 },
    { name: 'May', value: 51000 },
    { name: 'Jun', value: 53200 }];


  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Survey Button */}
      <a
        href="https://tally.so/r/w5JEBE"
        target="_blank"
        rel="noopener noreferrer"
        className="fixed top-1/2 -translate-y-1/2 right-[-28px] z-50 bg-blue-500 text-white py-2 px-5 rounded-t-lg shadow-lg flex items-center gap-2 transform -rotate-90 origin-bottom-right hover:right-[-10px] transition-all duration-200"
        title="Survey for the CivicMate"
      >
        <FileQuestion className="w-4 h-4" />
        <span className="font-semibold text-sm">Survey</span>
      </a>

      {/* Home Tab */}
      {activeTab === 'home' &&
        <div className="flex-1 overflow-y-auto pb-20">
          <div className="bg-blue-400 px-4 py-4">
            <div className="flex items-center justify-between h-12">
              <div className="h-full flex items-center">
                <img src="https://qtrypzzcjebvfcihiynt.supabase.co/storage/v1/object/public/base44-prod/public/160b96d6b_civic2-Edited.png" alt="Civic Mate Logo" className="h-8" />
              </div>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <div className="relative cursor-pointer">
                    <img
                      src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=100&h=100&fit=crop&crop=face"
                      alt="Profile"
                      className="w-8 h-8 rounded-full object-cover border-2 border-white" />

                    <div className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full flex items-center justify-center border-2 border-blue-400">
                      <span className="text-white text-xs font-bold">2</span>
                    </div>
                  </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-56" align="end">
                  <DropdownMenuLabel>My Account</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    <User className="mr-2 h-4 w-4" />
                    <span>Account</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Settings className="mr-2 h-4 w-4" />
                    <span>Settings</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <HelpCircle className="mr-2 h-4 w-4" />
                    <span>Support</span>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Log out</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          <div className="px-4 py-3 space-y-4 bg-gray-50 min-h-full">
            <div>
              <h2 className="text-gray-900 text-lg font-medium mb-3">Hi there,</h2>

              {/* Chatbot Input */}
              <form onSubmit={handleChatSubmit} className="mb-4">
                <div className="relative">
                  <input
                    type="text"
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    placeholder={isAgentThinking ? "AI is thinking..." : "Tell me what's happening..."}
                    disabled={isAgentThinking}
                    className="w-full px-4 py-3 pr-12 bg-white rounded-xl border border-gray-200 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent text-sm disabled:bg-gray-100"
                    onFocus={() => setSuggestionsVisible(true)}
                    onBlur={() => setSuggestionsVisible(false)}
                  />

                  <button
                    type="submit"
                    disabled={!chatMessage.trim() || isAgentThinking}
                    className="absolute right-3 top-1/2 transform -translate-y-1/2 text-blue-500 hover:text-blue-600 disabled:text-gray-300 transition-colors">
                    {isAgentThinking ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>

                  <AnimatePresence>
                    {suggestionsVisible && !activeEventContext && !isAgentThinking && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 10 }}
                        transition={{ duration: 0.2 }}
                        className="absolute top-full left-0 w-full mt-2 flex flex-wrap gap-2 z-10"
                      >
                        {quickSuggestions.map((suggestion, index) => (
                          <button
                            key={index}
                            type="button"
                            // Use onMouseDown to fire before the input's onBlur
                            onMouseDown={(e) => {
                              e.preventDefault(); // Prevents input from losing focus
                              handleSuggestionClick(suggestion.text);
                            }}
                            className="flex items-center gap-2 px-3 py-2 bg-white rounded-full border border-gray-200 text-sm text-gray-700 hover:bg-blue-50 hover:border-blue-300 transition-all duration-200 shadow-md hover:shadow-lg"
                          >
                            <span>{suggestion.icon}</span>
                            <span>{suggestion.text}</span>
                          </button>
                        ))}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </form>
            </div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">

              {/* Left Column - Checklist or Default Message */}
              <div>
                <AnimatePresence mode="wait">
                  {activeEventContext ?
                    <motion.div
                      key="checklist-panel"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ duration: 0.3 }}
                      className="bg-white rounded-xl p-4 shadow-md">

                      <div className="flex items-start gap-2 mb-4">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setActiveEventContext(null)}
                          className="text-gray-500 hover:text-gray-700 p-1">

                          <X className="w-4 h-4" />
                        </Button>
                        <div className="text-2xl">{LIFE_EVENTS[activeEventContext].headerIcon}</div>
                        <div>
                          <h3 className="font-bold text-base text-gray-900">{LIFE_EVENTS[activeEventContext].headerText}</h3>
                          <p className="text-xs text-gray-600">We have created a checklist to help you with the next steps.</p>
                        </div>
                      </div>

                      <div className="mb-4">
                        <div className="flex justify-between text-xs text-gray-500 mb-1">
                          <span>Progress</span>
                          <span>{completedTasks.size} of {LIFE_EVENTS[activeEventContext].tasks.length} completed</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div
                            className="bg-blue-500 h-1.5 rounded-full transition-all duration-500"
                            style={{ width: `${completedTasks.size / LIFE_EVENTS[activeEventContext].tasks.length * 100}%` }}>
                          </div>
                        </div>
                      </div>

                      <div className="space-y-2 max-h-80 overflow-y-auto no-scrollbar">
                        <AnimatePresence>
                          {LIFE_EVENTS[activeEventContext].tasks.map((task, index) => {
                            const isCompleted = completedTasks.has(index);
                            const isCurrent = !isCompleted && completedTasks.size === index;

                            if (isCompleted) {
                              return null; // Don't render completed tasks
                            }

                            const priority = LIFE_EVENTS[activeEventContext].taskPriorities[index] || 'medium';

                            const priorityColors = {
                              high: 'border-red-200 bg-red-50',
                              medium: 'border-yellow-200 bg-yellow-50',
                              low: 'border-blue-200 bg-blue-50'
                            };

                            return (
                              <motion.div
                                key={index}
                                className={`group relative rounded-lg p-3 border transition-all duration-300 cursor-pointer hover:shadow-sm ${
                                  isCurrent ?
                                    'border-blue-300 bg-blue-100' :
                                    priorityColors[priority] || 'border-gray-100 bg-gray-50'
                                  }`}
                                initial={{ opacity: 1, height: 'auto' }}
                                exit={{ opacity: 0, height: 0 }}
                                transition={{ duration: 0.3 }}
                                onClick={() => {
                                  if (!isCurrent) {
                                    toggleTaskComplete(index);
                                  }
                                }}
                              >
                                <div className="flex items-center gap-3">
                                  {/* Task Icon - Larger, no background */}
                                  <div className="w-10 h-10 flex items-center justify-center text-2xl">
                                    {TASK_ICONS[task] || '📄'}
                                  </div>

                                  {/* Content */}
                                  <div className="flex-1">
                                    <h4 className="font-medium text-sm text-gray-900">
                                      {task}
                                    </h4>

                                    {/* Status and Priority Badges */}
                                    <div className="flex items-center gap-1 mt-1">
                                      {isCurrent &&
                                        <span className="text-xs px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded-full font-medium">
                                          ● Current
                                        </span>
                                      }
                                      {!isCurrent &&
                                        <span className={`text-xs px-1.5 py-0.5 rounded-full font-medium ${
                                          priority === 'high' ?
                                            'bg-red-100 text-red-700' :
                                            priority === 'medium' ?
                                              'bg-yellow-100 text-yellow-700' :
                                              'bg-blue-100 text-blue-700'
                                          }`
                                        }>{priority} priority</span>
                                      }
                                    </div>
                                  </div>

                                  {/* Checkbox (Clicking the button will mark it complete) */}
                                  <div className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all duration-200 border-gray-300 group-hover:border-blue-400`}>
                                  </div>

                                  {/* Action Button for Current Task */}
                                  {isCurrent &&
                                    <Button
                                      size="sm"
                                      className="h-6 px-2 py-1 text-xs text-white bg-blue-500 hover:bg-blue-600"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        handleTaskClick(index); // Now opens the task modal
                                      }}>

                                      Start <ChevronRight className="w-3 h-3 ml-1" />
                                    </Button>
                                  }
                                </div>
                              </motion.div>);

                          })}
                        </AnimatePresence>
                      </div>

                      {/* Summary */}
                      {completedTasks.size > 0 &&
                        <div className="p-3 mt-4 border rounded-lg bg-blue-50 border-blue-200">
                          <div className="flex items-center gap-2">
                            <div className="flex items-center justify-center w-6 h-6 rounded-full bg-blue-500">
                              <Check className="w-3 h-3 text-white" />
                            </div>
                            <div>
                              <p className="text-sm font-medium text-blue-900">Great progress!</p>
                              <p className="text-xs text-blue-700">
                                You've completed {completedTasks.size} out of {LIFE_EVENTS[activeEventContext].tasks.length} tasks.
                              </p>
                            </div>
                          </div>
                        </div>
                      }
                    </motion.div> :

                    <motion.div
                      key="default-welcome-panel"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: 20 }}
                      transition={{ duration: 0.3 }}
                      className="bg-white rounded-xl p-4 shadow-md text-center flex flex-col items-center justify-center h-full min-h-[300px]">

                      <h3 className="font-bold text-lg text-gray-900 mb-2">What's happening in your life?</h3>
                      <p className="text-gray-600 mb-4">
                        Type into the chat above or select a Quick Action to get started.
                      </p>
                      <Settings className="w-16 h-16 text-gray-400 opacity-50 mb-4" />
                    </motion.div>
                  }
                </AnimatePresence>
              </div>

              {/* Right Column - Life Events (Quick Actions) */}
              <div>
                <h3 className="text-gray-900 text-lg font-medium mb-3">Quick Actions</h3>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(LIFE_EVENTS).map(([eventKey, event]) =>
                    <QuickActionCard key={eventKey} eventKey={eventKey} event={event} />
                  )}
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      }

      {/* Other tabs */}
      {activeTab === 'wallet' &&
        <div className="flex-1 overflow-y-auto pb-20 bg-gray-50">
          <div className="bg-blue-400 px-6 py-8">
            <h1 className="text-black text-3xl font-bold">Wallet</h1>
          </div>
          <div className="p-4 space-y-6">
            {/* Notification Banner */}
            <div className="p-4 rounded-xl shadow-sm bg-orange-100 border-l-4 border-orange-500">
              <div className="flex items-center gap-3">
                <div className="flex items-center justify-center w-8 h-8 rounded-full bg-orange-500">
                  <span className="text-lg text-white">⚠️</span>
                </div>
                <div>
                  <h3 className="font-semibold text-orange-900">Payment Due Soon</h3>
                  <p className="text-sm text-orange-700">You need to pay your rego in 2 days</p>
                </div>
              </div>
            </div>

            {/* Bank Account */}
            <div className="bg-white rounded-xl p-4 shadow-sm cursor-pointer hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gray-100">
                    <Landmark className="w-5 h-5 text-gray-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">Bank Accounts</h3>
                    <p className="text-xs text-gray-500">1 account linked</p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5 text-gray-400" />
              </div>
              <div className="flex justify-between items-baseline">
                <p className="text-sm text-gray-600">Available balance</p>
                <p className="text-2xl font-semibold text-gray-800">$2,450.78</p>
              </div>
            </div>

            {/* Superannuation */}
            <div className="bg-white rounded-xl p-4 shadow-sm cursor-pointer hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gray-100">
                    <CircleDollarSign className="w-5 h-5 text-gray-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">Superannuation</h3>
                    <p className="text-xs text-gray-500">AustralianSuper</p>
                  </div>
                </div>
                <ChevronRight className="w-5 h-5 text-gray-400" />
              </div>
              <div className="flex justify-between items-baseline">
                <p className="text-sm text-gray-600">Current balance</p>
                <p className="text-2xl font-semibold text-gray-800">$45,231.90</p>
              </div>
            </div>

            {/* Investment Portfolio */}
            <div className="bg-white rounded-xl p-4 shadow-sm">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gray-100">
                    <TrendingUp className="w-5 h-5 text-gray-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900">Investment Portfolio</h3>
                    <p className="text-xs text-gray-500">6-month performance</p>
                  </div>
                </div>
              </div>
              <div className="h-48 -ml-4">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={portfolioData} margin={{ top: 20, right: 20, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <Tooltip
                      contentStyle={{ borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '12px' }}
                      formatter={(value) => [`$${value.toLocaleString()}`, 'Value']}
                      labelFormatter={(label) => `End of ${label}`} />

                    <XAxis dataKey="name" tick={{ fontSize: 12 }} stroke="#9ca3af" axisLine={false} tickLine={false} />
                    <YAxis tickFormatter={(value) => `$${value / 1000}k`} tick={{ fontSize: 12 }} stroke="#9ca3af" axisLine={false} tickLine={false} />
                    <Area type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} fillOpacity={1} fill="url(#colorValue)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </div>
      }

      {activeTab === 'inbox' &&
        <div className="flex-1 overflow-y-auto pb-20 bg-gray-50">
          <div className="bg-blue-400 px-6 py-8">
            <h1 className="text-black text-3xl font-bold">Inbox</h1>
          </div>

          <div className="p-4">
            <div className="space-y-3">
              {inboxItems.map((item) =>
                <div
                  key={item.id}
                  className={`bg-white rounded-xl p-4 shadow-sm hover:shadow-md transition-shadow cursor-pointer border-l-4 ${
                    item.unread ? 'border-l-blue-500' : 'border-l-gray-200'
                    }`}>

                  <div className="flex items-start gap-3">
                    {/* Icon */}
                    <div className="flex items-center justify-center w-10 h-10 text-2xl">
                      {item.icon}
                    </div>

                    {/* Content */}
                    <div className="flex-1">
                      <div className="flex items-start justify-between">
                        <h3 className={`font-semibold text-sm leading-tight text-gray-900 ${
                          item.unread ? 'text-gray-900' : 'text-gray-700'
                          }`}>
                          {item.title}
                        </h3>
                        <span className="ml-2 text-xs text-gray-500">{item.time}</span>
                      </div>
                      <p className="mt-1 text-xs leading-relaxed text-gray-600">
                        {item.description}
                      </p>

                      {/* Unread indicator */}
                      {item.unread &&
                        <div className="flex items-center gap-1 mt-2">
                          <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                          <span className="text-xs font-medium text-blue-600">New</span>
                        </div>
                      }
                    </div>

                    {/* Arrow */}
                    <ChevronRight className="w-4 h-4 text-gray-400" />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      }

      {activeTab === 'services' &&
        <div className="flex-1 overflow-y-auto pb-20 bg-gray-50">
          <div className="bg-blue-400 px-6 py-8">
            <h1 className="text-black text-3xl font-bold">Services</h1>
          </div>

          <div className="p-6">
            <div className="grid grid-cols-2 gap-4">
              {servicesData.map((service, index) => {
                const Icon = service.icon;
                return (
                  <div
                    key={index}
                    className="flex flex-col justify-between p-4 bg-white rounded-2xl shadow-sm cursor-pointer hover:shadow-md transition-shadow">

                    <div>
                      <div className="flex items-center justify-center w-12 h-12 mb-4 rounded-xl bg-blue-100">
                        <Icon className="w-6 h-6 text-blue-600" strokeWidth={2} />
                      </div>
                      <p className="font-semibold leading-tight text-gray-900">{service.name}</p>
                    </div>
                    <div className="flex items-center justify-between mt-4">
                      <span className="flex items-center gap-1.5 text-xs font-medium text-green-600">
                        <span className="w-2 h-2 rounded-full bg-green-500"></span>
                        {service.status}
                      </span>
                      <ChevronRight className="w-4 h-4 text-gray-400" />
                    </div>
                  </div>);

              })}
            </div>
          </div>
        </div>
      }

      {/* Backdrop for FAB Menu */}
      <AnimatePresence>
        {isFabMenuOpen &&
          <motion.div
            className="fixed inset-0 bg-black/30 z-40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsFabMenuOpen(false)} />

        }
      </AnimatePresence>

      {/* Floating Action Button Menu (items) */}
      <div className="fixed bottom-[80px] left-1/2 -translate-x-1/2 z-50 flex flex-col items-center gap-3">
        <AnimatePresence>
          {isFabMenuOpen &&
            <motion.div
              className="flex flex-col-reverse items-center gap-3"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.2, staggerChildren: 0.06 }}>

              {[
                { label: 'Report Emergency', icon: Siren, action: () => setShowEmergencyModal(true) },
                { label: 'Show Personal ID', icon: User, action: () => setShowIdCardModal(true) },
                { label: 'Ask AI to describe', icon: Wand2, action: () => setShowVoiceModal(true) },
                { label: 'Report with Camera', icon: Camera, action: null, link: createPageUrl('ReportCamera') }].
                map((item, index) => {
                  const ButtonContent = () => (
                    <>
                      <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
                        <item.icon className="w-5 h-5 text-blue-500" />
                      </div>
                      <span className="text-sm font-medium">{item.label}</span>
                    </>
                  );

                  return (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 10 }}>
                      {item.link ? (
                        <Link to={item.link} className="bg-gray-900 text-white rounded-xl px-4 py-3 shadow-lg hover:bg-gray-800 transition-colors flex items-center gap-3 min-w-48">
                          <ButtonContent />
                        </Link>
                      ) : (
                        <button
                          className="bg-gray-900 text-white rounded-xl px-4 py-3 shadow-lg hover:bg-gray-800 transition-colors flex items-center gap-3 min-w-48"
                          onClick={() => { item.action(); setIsFabMenuOpen(false); }}>
                          <ButtonContent />
                        </button>
                      )}
                    </motion.div>
                  )
                }
                )}
            </motion.div>
          }
        </AnimatePresence>
      </div>

      {/* Bottom Navigation */}
      <div className="fixed bottom-0 left-0 right-0 bg-white/90 backdrop-blur-lg shadow-xl border-t border-gray-100 rounded-t-3xl">
        <div className="flex justify-around items-center h-16 px-4">
          {[
            { id: 'home', icon: Home, label: 'Home' },
            { id: 'wallet', icon: Wallet, label: 'Wallet' }].
            map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`group flex flex-col items-center justify-center flex-1 h-full transition-colors duration-200 focus:outline-none`}>

                  <div className={`
                    p-2 rounded-full transition-all duration-200
                    ${isActive ? 'bg-blue-100 text-blue-600' : 'text-gray-500 group-hover:bg-gray-100 group-hover:text-gray-700'}
                `}>
                    <Icon className="w-6 h-6" strokeWidth={2} />
                  </div>
                  <span className={`text-xs font-medium mt-1 ${isActive ? 'text-blue-600' : 'text-gray-500 group-hover:text-gray-700'}`}>{item.label}</span>
                </button>);

            })}

          {/* Center Microphone Button */}
          <div className="flex flex-col items-center justify-center flex-1 h-full">
            <Button
              size="icon"
              className="w-16 h-16 rounded-full bg-blue-400 shadow-lg text-white hover:bg-blue-500 transition-all duration-300 transform flex items-center justify-center"
              onClick={() => setIsFabMenuOpen(!isFabMenuOpen)}
              style={{ transform: isFabMenuOpen ? 'rotate(45deg) translateY(-16px)' : 'rotate(0deg) translateY(-16px)' }}>

              {isFabMenuOpen ? <X className="w-14 h-14" /> : <Compass className="w-14 h-14" />}
            </Button>
          </div>

          {[
            { id: 'inbox', icon: Mail, label: 'Inbox' },
            { id: 'services', icon: Grid3X3, label: 'Services' }].
            map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;

              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`group flex flex-col items-center justify-center flex-1 h-full transition-colors duration-200 focus:outline-none`}>

                  <div className={`
                    p-2 rounded-full transition-all duration-200
                    ${isActive ? 'bg-blue-100 text-blue-600' : 'text-gray-500 group-hover:bg-gray-100 group-hover:text-gray-700'}
                `}>
                    <Icon className="w-6 h-6" strokeWidth={2} />
                  </div>
                  <span className={`text-xs font-medium mt-1 ${isActive ? 'text-blue-600' : 'text-gray-500 group-hover:text-gray-700'}`}>{item.label}</span>
                </button>);

            })}
        </div>
      </div>

      {/* Life Event Details Modal */}
      {selectedEvent &&
        <Dialog open={!!selectedEvent} onOpenChange={() => {
          setSelectedEvent(null);
          setModalCompletedTasks(new Set()); // Reset modal tasks when closing
        }}>
          <DialogContent className="max-w-md mx-auto text-gray-900 bg-white border-gray-200">
            <DialogHeader>
              <div className="flex items-center gap-3 mb-4">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    setSelectedEvent(null);
                    setModalCompletedTasks(new Set());
                  }}
                  className="w-8 h-8 text-gray-500 hover:text-gray-700">

                  <X className="w-5 h-5" />
                </Button>
                <DialogTitle className="flex items-center gap-3 text-xl">
                  {React.createElement(LIFE_EVENTS[selectedEvent].icon, { className: "w-6 h-6" })}
                  {LIFE_EVENTS[selectedEvent].title}
                </DialogTitle>
              </div>
            </DialogHeader>

            <div className="space-y-6">
              <div className="text-center">
                <div className="w-32 h-32 mx-auto mb-4 overflow-hidden rounded-2xl border-2 border-gray-200">
                  <img
                    src={customImages[selectedEvent] || LIFE_EVENTS[selectedEvent].illustration}
                    alt="Event illustration"
                    className="object-cover w-full h-full" />

                </div>
                <p className="text-gray-600">{LIFE_EVENTS[selectedEvent].description}</p>
              </div>

              <div>
                <h4 className="mb-3 font-semibold">Tasks to complete:</h4>
                <div className="space-y-3 max-h-80 overflow-y-auto">
                  <AnimatePresence>
                    {LIFE_EVENTS[selectedEvent].tasks.map((task, index) => {
                      const isCompleted = modalCompletedTasks.has(index);

                      // Auto-hide completed tasks after a short delay
                      if (isCompleted) {
                        setTimeout(() => {
                          setModalCompletedTasks((prev) => {
                            const newSet = new Set(prev);
                            newSet.delete(index);
                            return newSet;
                          });
                        }, 1500);
                      }

                      return (
                        <AnimatePresence key={index}>
                          {!isCompleted &&
                            <motion.div
                              initial={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0, marginTop: 0, marginBottom: 0, paddingBottom: 0, paddingTop: 0 }} // Adjust exit styles to collapse
                              transition={{ duration: 0.3 }}
                              className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 cursor-pointer hover:bg-gray-100"
                              onClick={() => toggleModalTaskComplete(index)}>

                              <div className="text-2xl">{TASK_ICONS[task] || '📄'}</div>
                              <div className="flex-1">
                                <p className="font-medium text-gray-800">{task}</p>
                              </div>
                              <div className={`w-6 h-6 border-2 rounded flex items-center justify-center transition-all duration-200 ${
                                isCompleted ?
                                  'bg-green-500 border-green-500' :
                                  'border-gray-300 hover:border-blue-400'
                                }`
                                }>
                                {isCompleted &&
                                  <Check className="w-4 h-4 text-white" strokeWidth={3} />
                                }
                              </div>
                            </motion.div>
                          }
                        </AnimatePresence>);

                    })}
                  </AnimatePresence>
                </div>
              </div>

              <div className="flex gap-3">
                <Button
                  onClick={() => {
                    setSelectedEvent(null);
                    setModalCompletedTasks(new Set());
                    setActiveTab('services');
                  }}
                  className="flex-1 text-white bg-blue-500 hover:bg-blue-600">

                  Start Checklist
                </Button>
                <Button
                  onClick={() => {
                    setSelectedEvent(null);
                    setModalCompletedTasks(new Set());
                  }}
                  variant="outline"
                  className="flex-1">

                  Close
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      }

      {/* Task Details Modal */}
      {showTaskModal && selectedTaskIndex !== null && activeEventContext &&
        <Dialog open={showTaskModal} onOpenChange={() => {
          setShowTaskModal(false);
          setSelectedTaskIndex(null);
          setTaskFormData({ name: '', email: '', phone: '' });
        }}>
          <DialogContent className="max-w-md mx-auto text-gray-900 bg-white border-gray-200">
            <DialogHeader>
              <div className="flex items-center gap-3 mb-4">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    setShowTaskModal(false);
                    setSelectedTaskIndex(null);
                    setTaskFormData({ name: '', email: '', phone: '' });
                  }}
                  className="w-8 h-8 text-gray-500 hover:text-gray-700">

                  <X className="w-5 h-5" />
                </Button>
                <DialogTitle className="flex items-center gap-3 text-lg">
                  <div className="text-2xl">{TASK_ICONS[LIFE_EVENTS[activeEventContext].tasks[selectedTaskIndex]] || '📄'}</div>
                  {LIFE_EVENTS[activeEventContext].tasks[selectedTaskIndex]}
                </DialogTitle>
              </div>
            </DialogHeader>

            <div className="space-y-6">
              {/* Task Information */}
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <h4 className="font-semibold text-blue-900 mb-2">Service Information</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-blue-700">Department:</span>
                    <span className="text-blue-800 font-medium">Services Australia</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-700">Processing Time:</span>
                    <span className="text-blue-800 font-medium">1-3 business days</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-blue-700">Required Documents:</span>
                    <span className="text-blue-800 font-medium">ID verification</span>
                  </div>
                </div>
              </div>

              {/* Registration Form */}
              <form onSubmit={handleTaskSubmit} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Full Name *
                  </label>
                  <input
                    type="text"
                    required
                    value={taskFormData.name}
                    onChange={(e) => setTaskFormData({ ...taskFormData, name: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                    placeholder="Enter your full name" />

                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Email Address *
                  </label>
                  <input
                    type="email"
                    required
                    value={taskFormData.email}
                    onChange={(e) => setTaskFormData({ ...taskFormData, email: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                    placeholder="Enter your email" />

                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Phone Number
                  </label>
                  <input
                    type="tel"
                    value={taskFormData.phone}
                    onChange={(e) => setTaskFormData({ ...taskFormData, phone: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                    placeholder="Enter your phone number" />

                </div>

                <div className="flex gap-3 pt-4">
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => {
                      setShowTaskModal(false);
                      setSelectedTaskIndex(null);
                      setTaskFormData({ name: '', email: '', phone: '' });
                    }}
                    className="flex-1">

                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    className="flex-1 bg-blue-500 hover:bg-blue-600 text-white">

                    Submit Application
                  </Button>
                </div>
              </form>
            </div>
          </DialogContent>
        </Dialog>
      }

      {/* Simple Voice Modal */}
      {showVoiceModal &&
        <Dialog open={showVoiceModal} onOpenChange={() => setShowVoiceModal(false)}>
          <DialogContent className="max-w-md mx-auto text-gray-900 bg-white border-gray-200">
            <DialogHeader>
              <DialogTitle className="text-center">Voice Recording</DialogTitle>
            </DialogHeader>
            <div className="py-8 text-center">
              <div className="flex items-center justify-center w-20 h-20 mx-auto mb-4 rounded-full bg-purple-600">
                <Mic className="w-10 h-10 text-white" />
              </div>
              <p className="mb-4 text-gray-600">Voice recording feature coming soon</p>
              <Button onClick={() => setShowVoiceModal(false)}>Close</Button>
            </div>
          </DialogContent>
        </Dialog>
      }

      {/* Customize Modal */}
      <CustomizeImageModal
        show={showCustomizeModal}
        onClose={() => {
          setShowCustomizeModal(false);
          setCustomizeEventKey(null);
        }}
        eventKey={customizeEventKey}
        eventTitle={customizeEventKey ? LIFE_EVENTS[customizeEventKey]?.title : ''}
        currentImage={customizeEventKey ? customImages[customizeEventKey] || LIFE_EVENTS[customizeEventKey]?.illustration : ''}
        onImageUpdated={handleImageUpdated} />


      {/* Placeholder Modals for new FAB options */}
      {/* The Camera modal has been removed as it's now a navigation link */}
      <Dialog open={showIdCardModal} onOpenChange={setShowIdCardModal}>
        <DialogContent className="max-w-md mx-auto text-gray-900 bg-white border-gray-200">
          <DialogHeader><DialogTitle className="text-center">Personal ID Card</DialogTitle></DialogHeader>
          <div className="py-4 text-center text-gray-600">
            <div className="bg-blue-100 border border-blue-300 rounded-lg p-4">
              <h3 className="font-bold text-lg text-blue-900">John Citizen</h3>
              <p className="text-blue-700">ID: 123-456-789</p>
              <p className="text-sm text-blue-600 mt-2">Digital Identity Verified</p>
            </div>
          </div>
        </DialogContent>
      </Dialog>
      <Dialog open={showEmergencyModal} onOpenChange={setShowEmergencyModal}>
        <DialogContent className="max-w-md mx-auto text-gray-900 bg-white border-gray-200">
          <DialogHeader><DialogTitle className="text-red-600 text-center">Report Emergency</DialogTitle></DialogHeader>
          <p className="py-4 text-center text-gray-600">Emergency reporting feature is coming soon. In a real emergency, please call 000.</p>
        </DialogContent>
      </Dialog>
    </div>);
}
