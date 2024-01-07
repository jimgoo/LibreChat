import { useRecoilState } from 'recoil';
import type { ChangeEvent } from 'react';
import { useChatContext } from '~/Providers';
import { useRequiresKey, useFileHandling } from '~/hooks';
import AttachFile from './Files/AttachFile';
import StopButton from './StopButton';
import SendButton from './SendButton';
import Images from './Files/Images';
import Textarea from './Textarea';
import store from '~/store';
import React, { useState, useEffect, useCallback, useRef } from 'react';

let wsVideo: WebSocket | null = null;
let wsAudio: WebSocket | null = null;

export default function ChatForm({ index = 0 }) {
  const [text, setText] = useRecoilState(store.textByIndex(index));
  
  const {
    ask,
    files,
    setFiles,
    conversation,
    isSubmitting,
    handleStopGenerating,
    filesLoading,
    setFilesLoading,
    showStopButton,
    setShowStopButton,
  } = useChatContext();

  const { handleFiles } = useFileHandling();

  const BACKEND_URL = 'ws://localhost:8090';
  const BACKEND_URL_HTTP = 'http://localhost:8090';
  
  // references so that the websocket can access the latest values
  const conversationRef = useRef(conversation);
  const askRef = useRef(ask);
  const isSubmittingRef = useRef(isSubmitting);
  const handleFilesRef = useRef(handleFiles);
  const setFilesLoadingRef = useRef(setFilesLoading);
  const filesLoadingRef = useRef(filesLoading);
  const filesRef = useRef(files);

  useEffect(() => {
    conversationRef.current = conversation;
  }, [conversation]);

  useEffect(() => {
    askRef.current = ask;
  }, [ask]);

  useEffect(() => {
    isSubmittingRef.current = isSubmitting;
  }, [isSubmitting]);

  useEffect(() => {
    handleFilesRef.current = handleFiles;
  }, [handleFiles]);

  useEffect(() => {
    setFilesLoadingRef.current = setFilesLoading;
  }, [setFilesLoading]);

  useEffect(() => {
    filesLoadingRef.current = filesLoading;
  }, [filesLoading]);

  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  // useEffect to close the WebSocket when the component unmounts
  useEffect(() => {
    return () => {
      sendStopMessage();
      closeWebSocket(wsVideo);
      closeWebSocket(wsAudio);
    };
  }, []);

  const startWebsocket = useCallback((url: string, ws: WebSocket | null) => {
    try {

      ws = new WebSocket(url);

      ws.onopen = (event) => {
        console.log('ws.onopen', url);
        // ws.send(JSON.stringify(payload));
      };

      ws.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        
        // console.log('ws.onmessage: data', data);

        if (data.type === 'completion') {
          // wait until isSubmitting is false in case the model is still generating text
          while (isSubmittingRef.current) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          // console.log(`ws.onmessage: calling ask(${data.prompt}, tell: ${data.completion}), conversationId: ${conversationRef.current?.conversationId}, isSubmitting: ${isSubmittingRef.current}`);

          setText(data.prompt);
          //ask({ text: data.prompt, tell: data.completion });
          //ask({ text: data.prompt });
          askRef.current({ text: data.prompt });
          setText('');
        } else if (data.type === 'frames') {
          const files: File[] = [];
          for (const frameData of data.frames) {
            // console.log('ws.onmessage: frameData', typeof(frameData.frame), frameData.frame instanceof Blob);
            let base64String = frameData.frame;
            // Convert base64 to a byte array
            let byteCharacters = atob(base64String);
            let byteNumbers = new Array(byteCharacters.length);
            for (let i = 0; i < byteCharacters.length; i++) {
                byteNumbers[i] = byteCharacters.charCodeAt(i);
            }
            let byteArray = new Uint8Array(byteNumbers);
            const blob = new Blob([byteArray], { type: 'image/jpeg' });
            const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
            files.push(file);
          }
          if (files.length > 0) {
            // set loading state
            setFilesLoadingRef.current(true);
            // upload the images
            handleFilesRef.current(files);

            // wait until images are loaded before sending the next prompt
            // while (filesLoadingRef.current === true) { // does not work, always false
            while (true) {
              await new Promise(resolve => setTimeout(resolve, 100));
              // copy what's done in Images.tsx
              const filesArr = Array.from(filesRef.current.values());
              if (filesArr.every((file) => file.progress === 1)) {
                setFilesLoadingRef.current(false);
                break;
              }
            }

            setText(data.prompt);

            // wait until isSubmitting is false in case the model is still generating text
            while (isSubmittingRef.current) {
              await new Promise(resolve => setTimeout(resolve, 100));
            }

            // send prompt for the images
            askRef.current({ text: data.prompt });
            setText('');
          }
        } else if (data.type === 'transcript') {
          // console.log('ws.onmessage: transcript', data.prompt);
          setText(data.prompt);

          // wait until isSubmitting is false in case the model is still generating text
          while (isSubmittingRef.current) {
            await new Promise(resolve => setTimeout(resolve, 100));
          }

          askRef.current({ text: data.prompt });
          setText('');
        } else if (data.type === 'no-action') {
          if (data.message) {
            console.log('ws message: ', data.message);
          }
        } else {
          console.log('ws.onmessage: unknown data type', data.type);
        }
      }

      ws.onerror = (error) => {
        console.error(`ws.onerror:`, error);
        if (ws) ws.close();
      }

    } catch (error) {
      console.error('Error in websocket', error);
    }
  }, []);

  const sendStopMessage = async () => {
    console.log('sendStopMessage');
    // call HTTP post method 
    const response = await fetch(BACKEND_URL_HTTP + '/stop-stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      // <TODO> replace with actual user id
      body: JSON.stringify({'user_id': '6581b57ab264a19fbfdbc6cc' })
    });
    console.log('stop message response', response);
  }

  const closeWebSocket = (ws: WebSocket | null) => {
    if (ws) {
      ws.close();
      ws = null; // Set to null after closing
      console.log('Websocket closed');
    } else {
      console.log('Websocket already closed');
    }
  }

  const submitMessage = () => {
    // console.log('ChatForm: submitMessage: text: ', text, ', index: ', index, ', conversation: ', conversation);
    if (text.trim() === '/start') {
      startWebsocket(BACKEND_URL + '/ride-video', wsVideo);
      startWebsocket(BACKEND_URL + '/ride-audio', wsAudio);
    } else if (text.trim() === '/stop') {
      sendStopMessage();
      closeWebSocket(wsVideo);
      closeWebSocket(wsAudio);
    } else if (text.trim() === '/video') {
      startWebsocket(BACKEND_URL + '/ride-video', wsVideo);
    } else if (text.trim() === '/audio') {
      startWebsocket(BACKEND_URL + '/ride-audio', wsAudio);
    } else if (text.trim() === '/test') {
      startWebsocket(BACKEND_URL + '/ride-video-test', wsVideo);      
    } else {
      ask({ text });
    }
    setText('');
  };

  const { requiresKey } = useRequiresKey();

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        submitMessage();
      }}
      className="stretch mx-2 flex flex-row gap-3 last:mb-2 md:mx-4 md:last:mb-6 lg:mx-auto lg:max-w-2xl xl:max-w-3xl"
    >
      <div className="relative flex h-full flex-1 items-stretch md:flex-col">
        <div className="flex w-full items-center">
          <div className="[&:has(textarea:focus)]:border-token-border-xheavy border-token-border-heavy shadow-xs dark:shadow-xs relative flex w-full flex-grow flex-col overflow-hidden rounded-2xl border border-black/10 bg-white shadow-[0_0_0_2px_rgba(255,255,255,0.95)] dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:shadow-[0_0_0_2px_rgba(52,53,65,0.95)] [&:has(textarea:focus)]:shadow-[0_2px_6px_rgba(0,0,0,.05)]">
            <Images files={files} setFiles={setFiles} setFilesLoading={setFilesLoading} />
            <Textarea
              value={text}
              disabled={requiresKey}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setText(e.target.value)}
              setText={setText}
              submitMessage={submitMessage}
              endpoint={conversation?.endpoint}
            />
            <AttachFile endpoint={conversation?.endpoint ?? ''} disabled={requiresKey} />
            {isSubmitting && showStopButton ? (
              <StopButton stop={handleStopGenerating} setShowStopButton={setShowStopButton} />
            ) : (
              <SendButton text={text} disabled={filesLoading || isSubmitting || requiresKey} />
            )}
          </div>
        </div>
      </div>
    </form>
  );
}
