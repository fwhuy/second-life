import React from 'react';
import {AbsoluteFill, Composition, interpolate, Sequence, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {registerRoot} from 'remotion';
import {WebsiteShowcase} from './Showcase';

const palette = {ink:'#111714', green:'#1E3A2B', leaf:'#3E7C4F', cream:'#FAF7EF', tan:'#D8C7A8', orange:'#D97B29'};

const Title = ({kicker, children}) => {
  const frame = useCurrentFrame();
  const y = interpolate(frame, [0, 18], [32, 0], {extrapolateRight:'clamp'});
  const opacity = interpolate(frame, [0, 15], [0, 1], {extrapolateRight:'clamp'});
  return <div style={{position:'absolute',left:90,bottom:80,width:900,transform:`translateY(${y}px)`,opacity}}>
    <div style={{fontFamily:'Arial, sans-serif',fontSize:18,letterSpacing:7,color:palette.orange,fontWeight:800,marginBottom:16}}>{kicker}</div>
    <div style={{fontFamily:'Georgia, serif',fontSize:66,lineHeight:1.05,color:palette.cream,fontWeight:700}}>{children}</div>
  </div>;
};

const Cup = ({used=false}) => <div style={{position:'relative',width:230,height:300,transform:'rotate(-3deg)'}}>
  <div style={{position:'absolute',left:22,top:28,width:185,height:245,background:used?'#B79D78':palette.cream,clipPath:'polygon(5% 0,95% 0,83% 100%,17% 100%)',boxShadow:'0 30px 50px rgba(0,0,0,.28)'}} />
  <div style={{position:'absolute',left:8,top:8,width:215,height:36,borderRadius:18,background:palette.ink,border:`7px solid ${palette.tan}`}} />
  <div style={{position:'absolute',left:54,top:112,width:125,height:82,border:`4px solid ${palette.green}`,display:'grid',placeItems:'center',color:palette.green,font:'700 23px Georgia'}}>SECOND<br/>LIFE</div>
</div>;

const SceneOne = () => {
  const frame=useCurrentFrame(); const {fps}=useVideoConfig();
  const scale=spring({frame,fps,config:{damping:14}});
  return <AbsoluteFill style={{background:`radial-gradient(circle at 70% 35%, #486F59, ${palette.green} 45%, ${palette.ink})`}}>
    <div style={{position:'absolute',right:180,top:95,transform:`scale(${scale})`}}><Cup/></div>
    <Title kicker="A CAMPUS STORY">One brief day<br/>of a coffee cup.</Title>
  </AbsoluteFill>;
};

const SceneTwo = () => {
  const frame=useCurrentFrame();
  const x=interpolate(frame,[0,80],[220,-20],{extrapolateRight:'clamp'});
  return <AbsoluteFill style={{background:palette.cream,color:palette.ink,overflow:'hidden'}}>
    <div style={{position:'absolute',inset:0,backgroundImage:'linear-gradient(#3E7C4F18 1px,transparent 1px),linear-gradient(90deg,#3E7C4F18 1px,transparent 1px)',backgroundSize:'56px 56px'}}/>
    <div style={{position:'absolute',right:150,top:105,transform:`translateX(${x}px)`}}><Cup/></div>
    <div style={{position:'absolute',left:90,top:100,font:'800 18px Arial',letterSpacing:7,color:palette.orange}}>08:42 · FILLED</div>
    <div style={{position:'absolute',left:90,top:160,width:620,font:'700 62px Georgia',lineHeight:1.08,color:palette.green}}>Warmth for<br/>one morning.</div>
    <div style={{position:'absolute',left:90,bottom:80,width:710,font:'26px Arial',lineHeight:1.5,color:'#516257'}}>Made to be useful for minutes.<br/>Made from materials that last far longer.</div>
  </AbsoluteFill>;
};

const SceneThree = () => {
  const frame=useCurrentFrame();
  const rotate=interpolate(frame,[0,120],[0,8]);
  return <AbsoluteFill style={{background:`linear-gradient(135deg,#6D4D35,${palette.ink})`}}>
    <div style={{position:'absolute',right:170,top:100,transform:`rotate(${rotate}deg)`}}><Cup used/></div>
    <Title kicker="14:16 · THE HONEST END">Plastic lining changes<br/>where it belongs.</Title>
    <div style={{position:'absolute',right:70,bottom:54,font:'18px Arial',color:palette.tan}}>Most lined cups cannot enter ordinary paper recycling.</div>
  </AbsoluteFill>;
};

const SceneFour = () => {
  const frame=useCurrentFrame();
  const ring=interpolate(frame,[0,90],[.4,1],{extrapolateRight:'clamp'});
  return <AbsoluteFill style={{background:palette.green,display:'grid',placeItems:'center',textAlign:'center'}}>
    <div style={{position:'absolute',width:600,height:600,border:`4px solid ${palette.tan}`,borderRadius:'50%',transform:`scale(${ring})`,opacity:.45}}/>
    <div style={{zIndex:2}}>
      <div style={{font:'800 18px Arial',letterSpacing:8,color:'#BFD3C1'}}>THE BETTER ENDING</div>
      <div style={{font:'700 76px Georgia',lineHeight:1.08,color:palette.cream,margin:'28px 0'}}>Bring your own cup<br/>next time.</div>
      <div style={{font:'26px Arial',color:'#D9E4D6'}}>The best waste is the waste we never create.</div>
    </div>
  </AbsoluteFill>;
};

const Film = () => <AbsoluteFill style={{fontFamily:'Arial, sans-serif'}}>
  <Sequence durationInFrames={150}><SceneOne/></Sequence>
  <Sequence from={150} durationInFrames={180}><SceneTwo/></Sequence>
  <Sequence from={330} durationInFrames={180}><SceneThree/></Sequence>
  <Sequence from={510} durationInFrames={210}><SceneFour/></Sequence>
</AbsoluteFill>;

const Root = () => <>
  <Composition id="CoffeeCup" component={Film} durationInFrames={720} fps={30} width={1280} height={720}/>
  <Composition id="SecondLifeShowcase" component={WebsiteShowcase} durationInFrames={1800} fps={30} width={1280} height={720}/>
</>;
registerRoot(Root);
