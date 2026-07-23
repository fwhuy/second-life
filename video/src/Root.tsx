import { Composition } from "remotion";
import { SecondLifeVideo } from "./Video";
import { DURATION, FPS } from "./theme";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="SecondLifeAI"
      component={SecondLifeVideo}
      durationInFrames={DURATION}
      fps={FPS}
      width={1920}
      height={1080}
    />
  );
};
