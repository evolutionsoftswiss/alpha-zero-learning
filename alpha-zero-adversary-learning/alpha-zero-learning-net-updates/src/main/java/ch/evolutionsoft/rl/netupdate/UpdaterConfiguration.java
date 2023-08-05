package ch.evolutionsoft.rl.netupdate;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.support.PropertySourcesPlaceholderConfigurer;

@Configuration
@PropertySource(value = { "classpath:/application.properties" })
@ComponentScan("ch.evolutionsoft.rl.netupdate")
public class UpdaterConfiguration {

  @Value("${server.port}")
  private int adversaryLearningPort;

  @Bean
  public static PropertySourcesPlaceholderConfigurer propertySourcesPlaceholderConfigurer() {
     return new PropertySourcesPlaceholderConfigurer();
  }
  
  public int getAdversaryLearningPort() {
    return adversaryLearningPort;
  }

  public void setAdversaryLearningPort(int adversaryLearningPort) {
    this.adversaryLearningPort = adversaryLearningPort;
  }
}
