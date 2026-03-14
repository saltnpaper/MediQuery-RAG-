import { useState } from "react";
import {
  FileText,
  LogIn,
  MapPin,
  Plus,
  Send,
  ShieldCheck,
  Stethoscope,
  User,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";

type Message = {
  role: "assistant" | "user";
  content: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export default function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "Hi. I'm VitaChat, your AI assistant for understanding health insurance. I can explain coverage, compare plans, check claims, and pull answers from your healthcare RAG backend.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMsg: Message = { role: "user", content: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMsg.content }),
      });

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }

      const data = await response.json();
      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer ?? "No answer was returned by the backend.",
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `The backend request failed: ${message}. Make sure the FastAPI server is running on ${API_BASE_URL}.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const quickPrompt = (text: string) => {
    setInput(text);
  };

  return (
    <div className="flex h-screen bg-[#212121] text-white">
      <div className="hidden w-72 border-r border-[#2f2f2f] bg-[#171717] md:flex md:flex-col">
        <div className="flex items-center gap-3 border-b border-[#2f2f2f] p-5">
          <ShieldCheck className="text-green-400" />
          <h1 className="text-lg font-semibold">VitaChat</h1>
        </div>

        <div className="p-4">
          <Button className="flex w-full gap-2 bg-[#2f2f2f] hover:bg-[#3a3a3a]">
            <Plus size={16} /> New Chat
          </Button>
        </div>

        <div className="mt-4 px-4 text-xs text-gray-400">AI Tools</div>
        <div className="space-y-2 p-4 text-sm">
          <div className="flex cursor-pointer items-center gap-2 rounded p-2 hover:bg-[#2a2a2a]">
            <FileText size={16} />
            Check Coverage
          </div>
          <div className="flex cursor-pointer items-center gap-2 rounded p-2 hover:bg-[#2a2a2a]">
            <Stethoscope size={16} />
            Compare Plans
          </div>
          <div className="flex cursor-pointer items-center gap-2 rounded p-2 hover:bg-[#2a2a2a]">
            <MapPin size={16} />
            Find Nearby Care
          </div>
        </div>

        <div className="mt-4 px-4 text-xs text-gray-400">Recent</div>
        <div className="flex-1 space-y-2 p-4 text-sm text-gray-300">
          <div className="cursor-pointer rounded p-2 hover:bg-[#2a2a2a]">What is a deductible?</div>
          <div className="cursor-pointer rounded p-2 hover:bg-[#2a2a2a]">Does insurance cover therapy?</div>
          <div className="cursor-pointer rounded p-2 hover:bg-[#2a2a2a]">Urgent care vs ER</div>
        </div>

        <div className="space-y-3 border-t border-[#2f2f2f] p-4">
          <Button className="flex w-full items-center justify-center gap-2 bg-[#2f2f2f] hover:bg-[#3a3a3a]">
            <LogIn size={16} /> Log In
          </Button>
          <Button
            variant="outline"
            className="flex w-full items-center justify-center gap-2"
          >
            <User size={16} /> Sign Up
          </Button>
        </div>
      </div>

      <div className="flex flex-1 flex-col">
        <ScrollArea className="flex-1 p-4 md:p-10">
          <div className="mx-auto max-w-3xl space-y-6">
            {messages.map((msg, index) => (
              <div
                key={`${msg.role}-${index}`}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <Card
                  className={`max-w-xl ${
                    msg.role === "user"
                      ? "border-green-500 bg-green-600 text-white"
                      : "border-[#3a3a3a] bg-[#2a2a2a] text-gray-200"
                  }`}
                >
                  <CardContent className="p-4 text-sm leading-relaxed whitespace-pre-wrap">
                    {msg.content}
                  </CardContent>
                </Card>
              </div>
            ))}

            {loading && <div className="text-sm text-gray-400">VitaChat is thinking...</div>}
          </div>
        </ScrollArea>

        <div className="mx-auto flex max-w-3xl flex-wrap gap-2 px-4 pb-4">
          <button
            onClick={() => quickPrompt("What does my deductible mean?")}
            className="rounded bg-[#2a2a2a] px-3 py-2 text-xs hover:bg-[#333]"
          >
            What does my deductible mean?
          </button>
          <button
            onClick={() => quickPrompt("Does student insurance cover mental health?")}
            className="rounded bg-[#2a2a2a] px-3 py-2 text-xs hover:bg-[#333]"
          >
            Mental health coverage
          </button>
          <button
            onClick={() => quickPrompt("Should I go to urgent care or ER?")}
            className="rounded bg-[#2a2a2a] px-3 py-2 text-xs hover:bg-[#333]"
          >
            Urgent care vs ER
          </button>
        </div>

        <div className="border-t border-[#2f2f2f] bg-[#171717] p-4">
          <div className="mx-auto flex max-w-3xl gap-3">
            <Input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Ask about health insurance, coverage, claims..."
              className="border-[#3a3a3a] bg-[#212121]"
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  void sendMessage();
                }
              }}
            />

            <Button
              onClick={() => void sendMessage()}
              className="flex gap-2 bg-green-600 hover:bg-green-500"
            >
              <Send size={16} />
              Send
            </Button>
          </div>

          <p className="mt-3 text-center text-xs text-gray-500">
            VitaChat provides educational guidance about insurance. Always confirm details with your provider.
          </p>
        </div>
      </div>
    </div>
  );
}
