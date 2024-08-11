import { Avatar, Box, Divider, Grid, IconButton, LinearProgress, Paper, Stack, TextField, Typography } from '@mui/material';
import React from 'react';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import ChatMessage from './ChatMessage';
import Face5Icon from '@mui/icons-material/Face5';
import ReactTimeAgo from 'react-time-ago';
import useCallbackState from './useCallbackState';

interface MessageResponse {
    response: string
}
// Define the type for the props
interface MessageProps {
    messageData: ChatMessage
}

// Define the functional component using the props type
const Message: React.FC<MessageProps> = ({ messageData }) => {
    const alignment = messageData.is_robot ? "left" : "right"

    return (
        <Stack direction="row" sx={{ padding: 2, width: "100%", justifyContent: alignment }} alignContent={alignment}>
            {messageData.is_robot && <Avatar sx={{ marginRight: 2 }}>
                <SmartToyIcon />
            </Avatar>}
            <Stack direction="column">
                <Typography variant='h6'>{messageData.message}</Typography>
                <Typography variant='caption'><ReactTimeAgo date={messageData.date} locale="en-US" /></Typography>
            </Stack>
            {!messageData.is_robot && <Avatar sx={{ marginLeft: 2 }}>
                <Face5Icon />
            </Avatar>}
        </Stack>
    );
};


const Chat = () => {
    const [isLoading, setIsLoading] = React.useState<boolean>(false);
    const boxRef = React.useRef<HTMLDivElement>(null);
    const [messages, setMessages] = useCallbackState<Array<ChatMessage>>([
        {
            message: "Hello! How can I help you with your financial needs today?",
            is_robot: true,
            date: new Date()
        }
    ]);
    const [value, setValue] = React.useState<string>('');

    const loadResponse = (newMessages: ChatMessage[]) => {
        setIsLoading(true);

        fetch("/api/message/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                message: value,
            }),
        })
            .then(function (res) {
                return res.json();
            })
            .then(function (responseData: MessageResponse) {
                setMessages([...newMessages, {
                    message: responseData.response,
                    is_robot: true,
                    date: new Date()
                }]);
                setIsLoading(false);
            });
    };

    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setValue(event.target.value);
    };

    const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
        if (event.key === 'Enter') {
            addMessage();
        }
    };

    React.useEffect(() => {
        setValue('')
        handleScroll();
    }, [messages])

    const addMessage = () => {
        const newMessages: ChatMessage[] = [...messages, {
            message: value,
            is_robot: false,
            date: new Date()
        }];
        
        setMessages(newMessages, loadResponse);
    }

    const handleScroll = () => {
        if (boxRef.current) {
            boxRef.current.scrollTop = boxRef.current.scrollHeight;
        }
    };

    return (
        <Grid container component={Paper} sx={{ width: 'calc(100% - 20px)', height: '98vh', marginLeft: 1, marginRight: 1, marginTop: 1 }}>
            <Box sx={{ height: '95vh', width: "100%", display: 'flex', flexDirection: 'column' }}>
                <Box sx={{ flexGrow: 1, overflowY: "auto" }} ref={boxRef}>
                    <Stack sx={{ width: "100%" }}>
                        {messages.map((message) => <Message key={message.date.toISOString()} messageData={message} />)}
                    </Stack>
                </Box>

                <Box sx={{ height: '10px' }}>
                    {isLoading && <LinearProgress />}
                    <Divider />
                </Box>
                <Box sx={{ height: '90px' }}>
                    <Stack direction="row" sx={{ width: "100%", padding: 4 }}>
                        <TextField id="outlined-basic-email" value={value}
                            onChange={handleChange} onKeyUp={handleKeyPress} label="Type Something" fullWidth />
                        <IconButton aria-label="delete" onClick={addMessage}>
                            <SendIcon />
                        </IconButton>
                    </Stack>
                </Box>
            </Box>
        </Grid>
    );
}

export default Chat;
